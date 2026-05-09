"""Action Tools - Executable tools the ReAct agent can call.

Each tool makes a real DB mutation or simulated external call.
All tools return a dict with {success: bool, message: str, data: dict}.
"""

import uuid
from datetime import date, timedelta
from sqlalchemy import text
from src.core.database import SessionLocal
import structlog

log = structlog.get_logger()

# ── Tool registry (name → callable) ─────────────────────────────────────────
TOOLS: dict = {}


def register_tool(name: str):
    def decorator(fn):
        TOOLS[name] = fn
        return fn
    return decorator


def execute_tool(tool_name: str, tool_args: dict) -> dict:
    """Dispatch to the correct tool by name. Returns result dict."""
    if tool_name not in TOOLS:
        return {"success": False, "message": f"Unknown tool: {tool_name}", "data": {}}
    try:
        return TOOLS[tool_name](**tool_args)
    except TypeError as e:
        return {"success": False, "message": f"Invalid args for {tool_name}: {e}", "data": {}}
    except Exception as e:
        log.error("tool_execution_error", tool=tool_name, error=str(e))
        return {"success": False, "message": f"Tool error: {e}", "data": {}}


# ── Tool: create_po ──────────────────────────────────────────────────────────

@register_tool("create_po")
def create_po(product_id: int, qty: int, warehouse_id: int = 1) -> dict:
    """Create a Purchase Order for a product.

    Inserts into purchase_order + purchase_order_line.
    Returns the new PO number and ID.
    """
    with SessionLocal() as session:
        # Get supplier and unit price from product
        row = session.execute(text(
            "SELECT supplier_id, cost_price, name FROM product WHERE id = :pid"
        ), {"pid": product_id}).fetchone()
        if not row:
            return {"success": False, "message": f"Product {product_id} not found", "data": {}}

        supplier_id, cost_price, product_name = row
        if cost_price is None:
            cost_price = 0.0

        po_number = f"PO-{date.today().strftime('%Y%m%d')}-{str(uuid.uuid4())[:6].upper()}"
        total = float(cost_price) * qty
        expected = date.today() + timedelta(days=7)

        session.execute(text("""
            INSERT INTO purchase_order
                (po_number, supplier_id, warehouse_id, status, order_date, expected_delivery, total_amount, created_by)
            VALUES
                (:po_num, :sup_id, :wh_id, 'SUBMITTED', :order_date, :exp_date, :total, 'react_agent')
        """), {
            "po_num": po_number,
            "sup_id": supplier_id,
            "wh_id": warehouse_id,
            "order_date": date.today(),
            "exp_date": expected,
            "total": total,
        })

        # Get the PO id
        po_id = session.execute(text(
            "SELECT id FROM purchase_order WHERE po_number = :po_num"
        ), {"po_num": po_number}).scalar()

        # Get unit_price for line
        unit_price = session.execute(text(
            "SELECT cost_price FROM product WHERE id = :pid"
        ), {"pid": product_id}).scalar() or 0.0

        session.execute(text("""
            INSERT INTO purchase_order_line
                (purchase_order_id, product_id, quantity_ordered, quantity_received, unit_price)
            VALUES
                (:po_id, :prod_id, :qty, 0, :up)
        """), {"po_id": po_id, "prod_id": product_id, "qty": qty, "up": float(unit_price)})

        session.commit()

    log.info("create_po_done", po_number=po_number, product_id=product_id, qty=qty)
    return {
        "success": True,
        "message": f"Purchase Order {po_number} created for {qty} units of '{product_name}' (product #{product_id}). Total: ${total:,.2f}. Expected delivery: {expected}.",
        "data": {
            "po_number": po_number,
            "po_id": po_id,
            "product_id": product_id,
            "product_name": product_name,
            "qty": qty,
            "total_amount": total,
            "expected_delivery": str(expected),
        },
    }


# ── Tool: notify_supplier ────────────────────────────────────────────────────

@register_tool("notify_supplier")
def notify_supplier(supplier_id: int, message: str = "") -> dict:
    """Send a notification to a supplier (simulated HTTP POST).

    In production this would call an external email/webhook service.
    Here it logs the notification and records it in DB for audit.
    """
    with SessionLocal() as session:
        row = session.execute(text(
            "SELECT name, email, contact_name FROM supplier WHERE id = :sid"
        ), {"sid": supplier_id}).fetchone()
        if not row:
            return {"success": False, "message": f"Supplier {supplier_id} not found", "data": {}}
        supplier_name, email, contact = row

    # Simulate HTTP POST to supplier webhook
    notification_id = str(uuid.uuid4())[:8].upper()
    default_msg = f"You have a pending action required for your account (ref: {notification_id})"
    full_message = message or default_msg

    log.info(
        "supplier_notification_sent",
        supplier_id=supplier_id,
        supplier_name=supplier_name,
        email=email,
        notification_id=notification_id,
        simulated=True,
    )

    return {
        "success": True,
        "message": f"Notification sent to {supplier_name} ({email or 'no email on file'}). Contact: {contact or 'N/A'}. Notification ID: {notification_id}. [Simulated - production would POST to supplier portal]",
        "data": {
            "notification_id": notification_id,
            "supplier_id": supplier_id,
            "supplier_name": supplier_name,
            "email": email,
            "message_sent": full_message,
        },
    }


# ── Tool: update_shipment ────────────────────────────────────────────────────

_VALID_SHIPMENT_STATUSES = {"PENDING", "PROCESSING", "SHIPPED", "DELIVERED", "CANCELLED", "RETURNED"}


@register_tool("update_shipment")
def update_shipment(shipment_id: int, status: str) -> dict:
    """Update the status of an existing shipment record.

    Valid statuses: PENDING, PROCESSING, SHIPPED, DELIVERED, CANCELLED, RETURNED
    """
    status_upper = status.upper()
    if status_upper not in _VALID_SHIPMENT_STATUSES:
        return {
            "success": False,
            "message": f"Invalid status '{status}'. Valid: {', '.join(sorted(_VALID_SHIPMENT_STATUSES))}",
            "data": {},
        }

    with SessionLocal() as session:
        row = session.execute(text(
            "SELECT shipment_number, status FROM shipment WHERE id = :sid"
        ), {"sid": shipment_id}).fetchone()
        if not row:
            return {"success": False, "message": f"Shipment {shipment_id} not found", "data": {}}

        shipment_number, old_status = row

        # Set timestamp columns based on status
        ts_col = ""
        if status_upper == "SHIPPED":
            ts_col = ", shipped_date = NOW()"
        elif status_upper == "DELIVERED":
            ts_col = ", delivered_date = NOW()"

        session.execute(text(
            f"UPDATE shipment SET status = :status {ts_col} WHERE id = :sid"
        ), {"status": status_upper, "sid": shipment_id})
        session.commit()

    log.info("update_shipment_done", shipment_id=shipment_id, old=old_status, new=status_upper)
    return {
        "success": True,
        "message": f"Shipment {shipment_number} (ID: {shipment_id}) updated from '{old_status}' → '{status_upper}'.",
        "data": {
            "shipment_id": shipment_id,
            "shipment_number": shipment_number,
            "old_status": old_status,
            "new_status": status_upper,
        },
    }


# ── Tool: call_erp_sync ──────────────────────────────────────────────────────

@register_tool("call_erp_sync")
def call_erp_sync(order_ids: list, sync_type: str = "sales_orders") -> dict:
    """Trigger ERP synchronization for a list of order IDs.

    Simulates a microservice call to the ERP integration layer.
    In production this would POST to an internal microservice.
    """
    if not order_ids:
        return {"success": False, "message": "No order IDs provided", "data": {}}

    batch_id = f"ERP-SYNC-{str(uuid.uuid4())[:8].upper()}"

    # Simulate validation: check orders exist
    with SessionLocal() as session:
        table = "sales_order" if sync_type == "sales_orders" else "purchase_order"
        placeholders = ", ".join(f":id{i}" for i in range(len(order_ids)))
        params = {f"id{i}": oid for i, oid in enumerate(order_ids)}
        found = session.execute(text(
            f"SELECT id FROM {table} WHERE id IN ({placeholders})"
        ), params).fetchall()
        found_ids = [r[0] for r in found]
        missing = [oid for oid in order_ids if oid not in found_ids]

    log.info(
        "erp_sync_triggered",
        batch_id=batch_id,
        order_ids=order_ids,
        sync_type=sync_type,
        found=len(found_ids),
        missing=len(missing),
        simulated=True,
    )

    msg = f"ERP sync batch {batch_id} triggered for {len(found_ids)} {sync_type}."
    if missing:
        msg += f" Warning: {len(missing)} IDs not found: {missing}."
    msg += " [Simulated - production would POST to ERP microservice]"

    return {
        "success": True,
        "message": msg,
        "data": {
            "batch_id": batch_id,
            "sync_type": sync_type,
            "order_ids_requested": order_ids,
            "order_ids_found": found_ids,
            "order_ids_missing": missing,
            "status": "queued",
        },
    }


# ── Tool metadata (for LLM prompt) ──────────────────────────────────────────

TOOL_DESCRIPTIONS = {
    "create_po": {
        "description": "Create a purchase order for a product",
        "args": {
            "product_id": "int - the product ID to order",
            "qty": "int - quantity to order",
            "warehouse_id": "int (optional, default=1) - destination warehouse",
        },
        "example": '{"tool": "create_po", "args": {"product_id": 5, "qty": 100}}',
    },
    "notify_supplier": {
        "description": "Send a notification/alert to a supplier",
        "args": {
            "supplier_id": "int - the supplier ID to notify",
            "message": "str (optional) - custom message to send",
        },
        "example": '{"tool": "notify_supplier", "args": {"supplier_id": 3, "message": "Please confirm delivery date"}}',
    },
    "update_shipment": {
        "description": "Update the status of a shipment",
        "args": {
            "shipment_id": "int - the shipment ID to update",
            "status": "str - new status: PENDING|PROCESSING|SHIPPED|DELIVERED|CANCELLED|RETURNED",
        },
        "example": '{"tool": "update_shipment", "args": {"shipment_id": 12, "status": "SHIPPED"}}',
    },
    "call_erp_sync": {
        "description": "Trigger ERP synchronization for orders",
        "args": {
            "order_ids": "list[int] - list of order IDs to sync",
            "sync_type": 'str (optional, default="sales_orders") - "sales_orders" or "purchase_orders"',
        },
        "example": '{"tool": "call_erp_sync", "args": {"order_ids": [1, 2, 3]}}',
    },
}


def get_tools_prompt() -> str:
    """Build the tools section for the ReAct system prompt."""
    lines = ["Available tools:"]
    for name, info in TOOL_DESCRIPTIONS.items():
        lines.append(f"\n## {name}")
        lines.append(f"  Description: {info['description']}")
        lines.append("  Arguments:")
        for arg, desc in info["args"].items():
            lines.append(f"    - {arg}: {desc}")
        lines.append(f"  Example: {info['example']}")
    return "\n".join(lines)
