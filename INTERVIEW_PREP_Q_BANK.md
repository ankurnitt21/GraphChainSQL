# Interview Q&A bank — learning order

> **190 questions** — answers are written as if you are a **~5 YOE AI/ML engineer** in a real interview: direct structure, trade-offs, production instincts, and occasional "here's what I'd ship first." Questions follow a **recommended learning path** (foundations → retrieval → agents → production). Pair with `INTERVIEW_PREP.md` for GraphChainSQL-specific talking points.

## Recommended order (study top to bottom)

1. LangChain → 2. LangGraph → 3. Embeddings → 4. Vector DBs → 5. Reranking → 6. RAG → 7. Model routing → 8. Streaming → 9. AI agents → 10. ReAct → 11. Supervisor → 12. Multi-agent → 13. Scenarios → 14. Latency → 15. Cost/tokens → 16. Failures → 17. Testing → 18. Offline testing → 19. A/B testing.

**Sources (curated reading, not exhaustive):** industry interview roundups and guides such as [LockedIn AI — AI engineer interview questions](https://www.lockedinai.com/blog/ai-engineer-interview-questions), LangGraph/LangChain prep posts on LinkedIn, RAG question hubs (DataCamp, BuildML), and vector DB deep dives (HelloInterview). Use them for extra reading; answers here are standalone.

---

## 1 — LangChain

### Q1

**Question:** What is LangChain, and what problems does it solve?

**Answer:**

LangChain is a framework for composing LLM applications: prompts, model calls, output parsers,
retrievers, memory, and tool wiring become reusable building blocks instead of bespoke glue in
every repo.

The problem it solves is mostly engineering velocity and consistency—teams can standardize how
they do RAG, agents, and streaming, and swap providers (OpenAI, Anthropic, local) behind the
same interfaces. That matters when you iterate fast but still need tracing, batching, and async
patterns.

What I watch for in production is dependency surface and abstraction leaks: frameworks move
quickly, so I keep business logic thin, pin versions, and treat LangChain as orchestration—not
the place where domain rules hide.

### Q2

**Question:** What are the core LangChain components (prompts, models, parsers, retrievers, tools)?

**Answer:**

Prompts shape the instruction and inject variables safely; models are the actual inference
endpoints; parsers turn messy text into structured objects (JSON, pydantic) so downstream code
is deterministic.

Retrievers abstract “get relevant context” whether that is vector search, SQL, or hybrid; tools
expose side-effecting capabilities with schemas the model can target. In interview terms, that
is the full loop from user input → context → plan/action → validated output.

In mature systems I also treat observability as a first-class component: log prompt versions,
retrieval ids, and parser failures separately so you can tell whether the model, retrieval, or
post-processing broke.

### Q3

**Question:** How does LCEL work, and why is it useful vs older chains?

**Answer:**

LCEL is LangChain’s pipe-style composition: you chain callables with `|` so data flows left-to-
right, similar to Unix pipelines, but with support for streaming, async, batch, and parallel
execution.

Compared to older bespoke `Chain` subclasses, LCEL reduces API surface area: fewer magic
classes, more explicit wiring, and easier unit testing of each step in isolation.

The trade-off is readability for very large graphs—once you have branching, human approval, and
retries, I typically move orchestration to LangGraph while still using LangChain primitives
inside nodes.

### Q4

**Question:** How would you build a basic RAG pipeline with LangChain step by step?

**Answer:**

I start with ingestion: load documents (PDF/HTML/wiki), normalize text, then split with
structure-aware chunking when possible—headings and tables behave differently than plain
paragraphs.

Next I embed chunks, upsert into a vector store with stable ids and metadata (tenant, source
uri, timestamp). At query time I embed the question, retrieve top-k with filters, optionally
rerank, then stuff or compress context into a prompt template and call the model with strict
instructions about grounding and citations.

Only after the baseline works do I add evals (recall@k, faithfulness), caching, and guardrails.
Shipping order matters: observability and dataset first, fancy retrieval second.

### Q5

**Question:** What is the difference between a chain and an agent?

**Answer:**

A chain is a fixed program: the sequence of steps is known up front (retrieve → generate), which
makes testing, SLAs, and compliance easier because behavior is bounded.

An agent adds a control loop: the model chooses tools and next actions based on observations,
which unlocks flexible workflows but increases nondeterminism, token cost, and failure modes
like loops or wrong-tool arguments.

I pick chains when the task decomposes cleanly; I pick agents when the path depends on
intermediate results—but I still cap iterations, enforce allowlists, and log every tool call for
auditability.

### Q6

**Question:** How do you create and register a custom tool?

**Answer:**

I implement a small typed function with a clear docstring (the model reads that like an API
description), define arguments with a schema (pydantic or JSON schema), and register it with the
agent runtime so the model can bind tools or emit structured calls depending on the provider.

Registration is half the battle—validation is the other half: sanitize inputs, enforce authz
inside the tool, return structured errors the model can reason about, and never leak secrets in
error strings.

In code review I treat tool descriptions like UX copy: ambiguous names and vague parameter docs
are a leading cause of bad tool selection.

### Q7

**Question:** How does LangChain manage conversational memory, and what are trade-offs?

**Answer:**

Memory modules store prior turns—buffer (raw history), window (last N), summary (compress older
turns), or vector-backed memory for long sessions.

The trade-off is always tokens: more history improves coherence but raises latency, cost, and
the risk the model attends to irrelevant old content. In production I prefer summarization plus
retrieval over “dump everything,” and I keep a hard token budget per user.

I also version memory policies: changing summarization prompts silently can shift behavior, so I
log memory snapshots on escalations and support threads.

### Q8

**Question:** How do loaders, splitters, embeddings, and vector stores fit together?

**Answer:**

Loaders turn external sources into documents; splitters chunk them for embedding quality and
retrieval granularity; embeddings map chunks to vectors; vector stores index vectors for fast
approximate search.

That pipeline is where a lot of RAG quality is actually won or lost—bad chunking cannot be fully
fixed by a better LLM. I tune chunk size/overlap against labeled queries, not intuition alone.

Operationally I track embedding model version and index lag: if re-embedding lags writes, users
see stale answers even when the UI looks healthy.

### Q9

**Question:** How do you debug and observe LangChain apps in production?

**Answer:**

I use distributed tracing (OpenTelemetry) and/or LangSmith-style run tracing: one trace per user
request with child spans for retrieve, rerank, each LLM call, and tool execution.

Logs should be structured: prompt id/version, model name, temperature, retrieved chunk ids,
latency per step, token usage, and parser outcomes. That lets you separate retrieval misses from
instruction violations.

For debugging I keep golden queries and replay them when prompts or models change—subjective
“vibes” don’t scale when the team grows.

### Q10

**Question:** When use LangChain alone vs add LangGraph?

**Answer:**

LangChain alone is great for linear pipelines: classic RAG, simple tool use, LCEL streaming.

I add LangGraph when I need cycles, branching, durable checkpoints, human-in-the-loop
interrupts, or explicit multi-agent routing—basically anything that is a state machine, not a
straight line.

Practically I also consider team readability: graphs make control flow visible in code review,
which matters for regulated workflows and on-call playbooks.

---

## 2 — LangGraph

### Q1

**Question:** What is LangGraph, and how is it different from LangChain?

**Answer:**

LangGraph models an application as a graph: nodes are steps, edges are transitions, and shared
typed state flows through the system with optional persistence.

LangChain/LCEL excels at composing linear transformations; LangGraph excels when the control
structure itself is part of the product—agents that may revisit steps, pause for humans, or
route based on runtime signals.

If you only need retrieve-then-generate, LangGraph is probably overkill; if you need
resumability and explicit policies, LangGraph pays off quickly.

### Q2

**Question:** Why is graph orchestration useful in production agents?

**Answer:**

Production agents need explicit policies: when to validate SQL, when to ask a human, when to
stop spending tokens, and how to retry safely. A graph encodes those rules as first-class
structure instead of burying them in prompt prose.

It also improves operability: on-call can reason about “which node failed” from traces aligned
to graph nodes, and you can attach SLOs per node (retrieval p95 vs generation p95).

Finally, graphs make testing easier: you can unit test a node’s pure function contract and
integration test transitions with mocked LLM outputs.

### Q3

**Question:** How does state management work in LangGraph?

**Answer:**

You define a state schema (often dict-like) that accumulates fields across the run—messages,
retrieved docs, tool results, flags like `needs_human`, and intermediate artifacts.

Each node returns a partial update that merges into state according to reducers—so you can
append messages, merge dicts, or apply domain-specific merge rules.

The key discipline is minimal shared surface: pass summaries and references between nodes
instead of copying huge histories everywhere, otherwise you blow token budgets and create
inconsistent views.

### Q4

**Question:** What are nodes, edges, and conditional edges?

**Answer:**

Nodes are functions or callables that read state and emit updates—often wrapping an LLM call, a
tool, or deterministic validation.

Edges define default transitions after a node completes. Conditional edges route based on state:
for example route to `human_review` if confidence is low, else continue.

I use conditional edges for anything that would otherwise be nested `if` statements around LLM
calls—keeping control flow explicit reduces “prompt spaghetti.”

### Q5

**Question:** How do reducers help when multiple branches update shared state?

**Answer:**

When you fan out in parallel, two branches might update the same key; reducers define how to
merge safely —append-only lists, max of scores, or last-write-wins where appropriate.

Without reducers you get race-y updates and lost messages—especially painful under load when
timing changes.

I pick reducers based on semantics: for chat history, append; for monotonic progress counters,
max; for single authoritative decisions, explicit locks or a supervisor merge step.

### Q6

**Question:** What is checkpointing, and why does it matter?

**Answer:**

Checkpointing persists state after steps so a run can resume after crashes, deploys, or human
approval. It turns an agent from a fragile request/response loop into a durable workflow.

That matters for human-in-the-loop, long-running research agents, and anything where you cannot
afford to repeat expensive tool side effects.

Operationally I always isolate checkpoints by `thread_id` / user id and encrypt at rest if state
can contain PII or retrieved documents.

### Q7

**Question:** How do you implement human-in-the-loop approval?

**Answer:**

I pause the graph at a designated node when a policy triggers—large money movement, low
confidence, or sensitive data access—persist pending state to the checkpointer, and expose an
API/UI for approve/reject.

On approval, the graph resumes from the saved checkpoint with an explicit human decision
recorded in state for audit trails.

I avoid “human in the prompt only” patterns; the pause must be enforced by orchestration, not
something the model can talk its way past.

### Q8

**Question:** What are subgraphs, and when would you use them?

**Answer:**

A subgraph is a reusable graph embedded in a parent graph—think a packaged ReAct loop or a SQL
generation/repair micro-pipeline.

They help modularity: teams can own a subgraph contract (inputs/outputs) and test it
independently.

I use subgraphs when the same pattern repeats across products (support triage, code patch
proposals) but the surrounding business flow differs.

### Q9

**Question:** How do you design a multi-agent system in LangGraph?

**Answer:**

I start with roles and interfaces: each agent is a node or subgraph with narrow tools and a
clear mandate. A supervisor or router node decides delegation based on rules, a classifier, or
an LLM router with guardrails.

Shared state should be minimal: pass task specs, partial results, and citations—not full chat
logs to every agent.

Observability-wise I tag spans per agent and store handoff decisions; debugging multi-agent
without that is basically reading novels in logs.

### Q10

**Question:** What are common LangGraph traps (loops, routing, persistence)?

**Answer:**

Infinite loops when stop conditions are unclear or max-iterations is too high; flaky routing
when prompts are ambiguous; missing checkpoint configuration so resumes are impossible.

Another trap is poor `thread_id` hygiene—mixing users’ state is a security incident waiting to
happen.

I mitigate with explicit caps, structured routing outputs (JSON), integration tests for
transitions, and checkpointer TTL/retention policies aligned to compliance.

---

## 3 — Embeddings

### Q1

**Question:** What are embeddings, and why do they matter for semantic search and RAG?

**Answer:**

Embeddings map text (or multimodal inputs) into dense vectors where semantic similarity
corresponds to geometric nearness—so “car” and “automobile” cluster closer than “car” and
“banana.”

That enables retrieval beyond keywords: paraphrases, concepts, and cross-language matches—core
to RAG when users don’t phrase queries like your documentation.

They are not magic: quality depends on domain fit, chunking, and evaluation—embeddings are a
retrieval subsystem, not a substitute for product design.

### Q2

**Question:** Explain embeddings to a non-technical interviewer.

**Answer:**

I describe it as turning sentences into coordinates in a high-dimensional map: similar ideas
land near each other, so search becomes “find the nearest points.”

Then I connect it to user value: customers can ask questions naturally and we pull the right
policy snippets or runbooks without perfect keyword matches.

I keep the analogy honest: approximate neighbors can be wrong, which is why we add filters,
reranking, and human review for risky domains.

### Q3

**Question:** Keyword matching vs embedding retrieval?

**Answer:**

Keyword/BM25 is great for exact tokens, rare strings, SKUs, and error codes; dense embeddings
excel at paraphrase and conceptual similarity.

In production I often use hybrid retrieval: fuse dense and sparse results (RRF or learned
fusion) so you don’t lose precision on identifiers while keeping semantic recall.

The decision is empirical: if your users type jargon that appears verbatim in docs,
underweighting keywords hurts; if they speak in natural language, dense-only misses nuance.

### Q4

**Question:** How do cosine similarity and distance relate?

**Answer:**

Cosine similarity measures angle between vectors (scale-invariant if norms vary); L2 distance
measures Euclidean separation in the embedding space.

For normalized embeddings, cosine and Euclidean distances are monotonically related in useful
regimes—but mixing metrics across systems without consistency breaks ANN indexes tuned for one
assumption.

Operationally I pick one convention for the index, document it, and ensure training/serving use
the same normalization pipeline.

### Q5

**Question:** What factors matter when choosing an embedding model?

**Answer:**

Domain performance (legal vs code vs support), dimensionality (memory/latency), multilingual
needs, max sequence length, throughput, licensing, and whether you need local deployment for
privacy.

I also look at stability: frequent silent upgrades can shift retrieval behavior, so I pin
versions and re-run offline evals when upgrading.

Cost matters at scale: embedding every chunk on every reindex can dominate spend if the model is
huge—sometimes a smaller embedding model plus reranking wins overall.

### Q6

**Question:** How do embedding models affect downstream retrieval quality?

**Answer:**

If embeddings misrepresent domain language, you retrieve the wrong neighbors—then even a
frontier LLM cannot answer faithfully because the context is wrong.

That is why I measure end-to-end QA and citation correctness, not only embedding leaderboard
scores.

When retrieval fails, I triage: query distribution shift, chunking issues, stale index, or
embedding mismatch—each has different fixes.

### Q7

**Question:** What if indexing and query use different embedding models?

**Answer:**

That is a serious bug class: vectors from different spaces are not comparable, so neighbors
become random-ish relative to intent.

The fix is to freeze one model end-to-end or run a controlled re-embedding migration with dual
indexes and shadow testing.

I’ve seen this happen when one team upgrades the query embedder but not the corpus—guard with CI
tests that assert model ids match in ingest and query paths.

### Q8

**Question:** How do chunk size and overlap affect embeddings?

**Answer:**

Too-small chunks lose necessary context; too-large chunks dilute specificity so embeddings
become “averages” of unrelated ideas.

Overlap reduces boundary cuts mid-entity and helps continuity, but increases storage and
duplicate neighbors if not deduped carefully.

I tune chunking using labeled retrieval tasks on real documents—especially tables, lists, and
nested headings where naive splits fail.

### Q9

**Question:** Common failure modes in domain-specific search?

**Answer:**

Out-of-domain jargon the embedder never saw, acronym collisions, stale corpora, duplicated
documents skewing neighbors, and language mismatch between users and docs.

Another subtle one is “attractive” wrong answers: fluent retrieved text that looks relevant but
is not authoritative.

Mitigations include metadata filters, freshness signals, dedupe, source ranking, and human
feedback loops on bad retrievals.

### Q10

**Question:** How evaluate a new embedding model vs current?

**Answer:**

I run the same retrieval + QA pipeline on a frozen labeled set: recall@k, MRR/nDCG for ranking,
plus downstream answer correctness and refusal behavior.

I also check tail latency and cost because a better model that breaks SLOs is not deployable.

Finally I look for regressions on adversarial queries—exact match cases that hybrid/BM25 used to
save.

---

## 4 — Vector databases

### Q1

**Question:** What is a vector database vs a relational DB?

**Answer:**

Relational databases optimize transactional row storage, joins, and constraints; vector
databases optimize approximate nearest-neighbor search in high dimensions plus metadata
filtering.

You can store vectors in Postgres with pgvector—and many teams do—but “vector DB” often implies
ANN indexes, sharding, and serving patterns tuned for similarity workloads.

In hybrid architectures I still use OLTP for source-of-truth rows and a vector index for
semantic retrieval, linked by stable ids.

### Q2

**Question:** Why are high-dimensional vectors hard to search exactly?

**Answer:**

Exact nearest neighbor degrades with dimensionality and data scale—brute force is linear per
query and too slow for online serving at millions of vectors.

ANN algorithms (HNSW, IVF, PQ variants) trade a small amount of recall for orders-of-magnitude
speedups.

The interview point is knowing the trade-off: you must monitor recall on a canary set, not
assume ANN is “good enough” forever.

### Q3

**Question:** What is ANN search, and why use it?

**Answer:**

ANN returns likely nearest neighbors fast by pruning the search space using graph or inverted-
file structures rather than scanning everything.

It is used because product SLAs require tens to hundreds of milliseconds, not seconds, for
retrieval at scale.

Tuning parameters (ef, nprobe, quantization) is an ops + ML problem: latency vs recall vs
memory.

### Q4

**Question:** Common use cases for vector DBs in AI?

**Answer:**

RAG chunk retrieval, semantic cache, deduplication, support ticket similarity, fraud clustering,
and recommendations—anywhere “find similar” is core.

They also show up in eval tooling: nearest neighbor audits for toxic or sensitive content.

I always pair vector search with metadata filters for multi-tenant isolation and ACL
enforcement.

### Q5

**Question:** How do indexing strategies affect recall, speed, and cost?

**Answer:**

More aggressive quantization reduces RAM and cost but can hurt recall; deeper graph search
improves recall at higher latency.

Sharding and replication change tail latency under load—p99 matters more than p50 for user-
facing chat.

I treat index parameters like hyperparameters: measure on real queries, not synthetic random
vectors.

### Q6

**Question:** What metadata filtering should a good vector DB support?

**Answer:**

Tenant id, document type, time range, language, permission tags, and source system
identifiers—anything you’d put in a WHERE clause conceptually.

Filtering is not optional for enterprise RAG: without it you risk cross-tenant leakage or
retrieving draft vs published content.

I design schemas so filters are selective and indexed; over-broad filters that still scan huge
partitions hurt latency.

### Q7

**Question:** How choose Pinecone vs Weaviate vs Chroma vs FAISS?

**Answer:**

Pinecone is managed SaaS with low ops overhead; Weaviate offers rich filtering and hybrid
features self-hosted or managed; Chroma is great for local dev and smaller apps; FAISS is an in-
process library when you own the whole serving stack.

Decision drivers are ops maturity, multi-tenant isolation, hybrid search needs, latency SLOs,
backup/restore, and compliance (region, VPC).

I avoid picking purely on benchmarks—integration cost and on-call burden dominate at 5 years
experience.

### Q8

**Question:** What happens when data is noisy, duplicated, or poorly chunked?

**Answer:**

Neighbors become junk: the model sees plausible-sounding but wrong context, and rerankers cannot
invent missing gold documents.

Duplicates inflate certain chunks’ retrieval frequency and distort ranking; bad chunking splits
tables and splits entities across boundaries.

Fix data first: dedupe, cleaning pipelines, structure-aware chunking—then revisit embedding
models.

### Q9

**Question:** How handle updates, deletes, and re-embedding at scale?

**Answer:**

I version embeddings and maintain idempotent jobs: upsert vectors with stable chunk ids,
tombstone deletes, and track index lag metrics.

During migrations I dual-write or shadow-read to compare recall before cutover.

Backpressure matters: if ingest spikes, you don’t want embedding queues to silently grow until
retrieval is hours stale.

### Q10

**Question:** What metrics monitor in production vector retrieval?

**Answer:**

Latency p95/p99, error rates, QPS, index freshness lag, embedding job failures, ANN recall on a
labeled canary set, and cost per query.

I also monitor empty-result rates and filter selectivity—sudden spikes often indicate bad
deploys or schema mistakes.

Business metrics like escalation rate complement technical metrics for RAG quality.

---

## 5 — Reranking

### Q1

**Question:** What is reranking, and where does it fit?

**Answer:**

Reranking is a second-stage scorer: you retrieve a wider candidate set cheaply, then score
query–document pairs with a more expressive model before stuffing context.

It sits after retrieval and before generation—tightening precision so the LLM sees fewer, better
chunks.

It is especially valuable when bi-encoder retrieval is noisy but you can afford extra
milliseconds or a few cents per query.

### Q2

**Question:** Why can reranking beat raw vector similarity?

**Answer:**

Bi-encoders embed query and document independently, which is fast but cannot capture all fine-
grained interactions. Cross-attention rerankers jointly read query and document and can detect
subtle relevance.

That helps on multi-constraint questions like “this error on this OS version after this
upgrade.”

The trade-off is compute: cross-encoders scale poorly with very long documents, so you often
rerank short chunk summaries or headline fields.

### Q3

**Question:** Bi-encoder vs cross-encoder?

**Answer:**

Bi-encoder: embed separately, approximate nearest neighbors—great for stage one at scale.

Cross-encoder: attends across tokens from both sides—more accurate, slower, typically used on
dozens to hundreds of candidates, not millions.

Some systems use a small cross-encoder for first rerank and an LLM judge only on high-risk
queries.

### Q4

**Question:** When is reranking worth extra latency and cost?

**Answer:**

When precision dominates recall in business outcomes—legal, medical, finance, or high-stakes
support—or when vector retrieval frequently pulls “almost right” distractors.

It is less compelling when latency budgets are tight and retrieval is already high-recall on a
golden set.

I justify with offline lift on MRR/answer correctness and online A/B on escalations—not vibes.

### Q5

**Question:** How many chunks into a typical reranker?

**Answer:**

Common ranges are tens to a few hundred candidates narrowed to single digits to teens for
context stuffing, depending on model limits and latency.

The exact numbers come from measuring marginal gains: doubling candidates helps until reranker
noise dominates.

I also cap document length per candidate to keep reranker latency predictable.

### Q6

**Question:** How evaluate rerankers (MRR, MAP, nDCG)?

**Answer:**

With labeled relevance judgments per query–doc pair, I compare ranking metrics before vs after
rerank and correlate with downstream task success.

Pointwise accuracy alone can mislead if the candidate pool is biased—always evaluate end-to-end.

For production, I track reranker-specific failure slices: queries where the gold doc was
retrieved but reranked out of top-n.

### Q7

**Question:** Which queries benefit most from reranking?

**Answer:**

Ambiguous wording, multi-hop constraints, domains with many near-duplicates, and cases where
vector similarity confuses related policies.

Conversely, exact SKU lookups often benefit more from lexical retrieval than reranking alone.

Intent routing can skip reranking for “easy” queries to save cost—dynamic policies beat always-
on rerank.

### Q8

**Question:** How can reranking still fail if the gold doc is in the candidate set?

**Answer:**

The reranker can mis-score long contexts, key evidence may be buried below truncation, or labels
may not match user intent (annotation mismatch).

Sometimes the chunk is technically relevant but not authoritative—rerankers can favor fluent
noise.

Debugging requires inspecting pairwise scores and chunk boundaries, not assuming rerank fixes
retrieval.

### Q9

**Question:** Combine metadata filters, hybrid search, and reranking?

**Answer:**

I filter first to enforce ACLs and shrink space, run hybrid retrieval to fuse sparse+dense
candidates, union/dedupe, then rerank the merged pool.

Order matters: reranking millions is infeasible; filters and ANN are there to build a reasonable
candidate set.

I log each stage’s contribution so you can see whether failures are filter-too-tight vs
retrieval vs rerank.

### Q10

**Question:** Justify reranking to a latency-focused team?

**Answer:**

I show measured impact: reduced human review minutes, fewer bad answers, higher
deflection—converted into dollars—versus p95 latency increase.

Mitigations include reranking only premium tiers, async rerank for non-chat surfaces, caching
rerank scores for repeated queries, and smaller cross-encoders.

Framing it as a precision stage with an explicit SLA budget usually lands better than “ML says
so.”

---

## 6 — RAG

### Q1

**Question:** What is RAG, and how does it improve plain LLM generation?

**Answer:**

RAG retrieves grounding passages from your knowledge base and conditions generation on them, so
answers can cite internal policies and fresher facts without retraining the base model.

It reduces pure hallucination risk in factual domains—though it does not eliminate fabrication
if the model ignores context or the retrieval is wrong.

I position RAG as a systems architecture: retrieval quality, permissions, and evals matter as
much as the LLM choice.

### Q2

**Question:** Walk through the full RAG pipeline.

**Answer:**

Ingestion and normalization, cleaning, structure-aware chunking, embedding, indexing with
metadata, then query-time retrieval with filters, optional rerank, context assembly
(stuff/compress), generation, and post-validation (citations, format).

Around that I add observability, cost controls, and feedback capture—otherwise you cannot
iterate safely.

For regulated settings I also log provenance and support redaction/retention policies on stored
chunks.

### Q3

**Question:** How decide chunk size, overlap, and splitting?

**Answer:**

Start from the downstream task: what is the smallest self-contained unit a model needs to answer
correctly? Then choose chunk size and overlap to preserve continuity without bloating the index.

Structure-aware splits (by heading, section, table rows) usually beat fixed character counts on
enterprise PDFs.

I validate empirically with recall@k and human rubrics; chunking is a hyperparameter tied to
your document distribution.

### Q4

**Question:** What is hybrid retrieval, and when better than dense alone?

**Answer:**

Hybrid combines lexical signals (BM25) with dense embeddings, then fuses results—common fusion
is RRF or weighted linear combination.

It wins when queries contain rare tokens, SKUs, or exact legal language where embeddings alone
miss.

Cost is complexity: two retrieval paths to maintain, tune, and monitor—worth it when precision
failures are expensive.

### Q5

**Question:** Diagnose bad RAG: retrieval failure vs generation failure?

**Answer:**

If the annotated gold chunk is not in top-k, that is primarily retrieval—fix chunking,
embeddings, filters, hybrid, or rerank.

If gold is present but the answer ignores it, that is generation/instruction-following—tighten
prompts, reduce context noise, reorder chunks, or add extraction steps.

I always trace both with chunk ids in logs; teams often argue about the model when retrieval
never fetched the right paragraph.

### Q6

**Question:** Best ways to reduce hallucinations in RAG?

**Answer:**

Improve retrieval precision, require citations to specific chunks, constrain answers to context-
only, add deterministic validators (regex/grammar), and use a verifier pass for high-risk
outputs.

Operational hygiene matters: stale indexes and wrong-tenant retrieval cause confident wrong
answers.

I also measure faithfulness with automated judges plus spot human review—fluency is the wrong
metric.

### Q7

**Question:** Evaluate RAG beyond exact match?

**Answer:**

Task success, human rubrics (correctness, completeness, safety), citation accuracy, faithfulness
scores, latency, cost, and regression suites on golden queries.

For open-ended answers I use LLM-as-judge carefully: calibrate against humans and watch judge
drift when models change.

Online metrics like thumbs-down, escalation rate, and rework time connect evals to business
outcomes.

### Q8

**Question:** Handle multi-hop RAG?

**Answer:**

Iterative retrieve-read-retrieve, query decomposition into subquestions, or graph-based
retrieval over entities and relations—each encodes different assumptions about the knowledge
structure.

I always cap steps and spend: multi-hop agents can explode tokens and tool calls.

Debugging multi-hop needs intermediate artifacts in traces so you can see each hop’s evidence.

### Q9

**Question:** Common production failure modes in naive RAG?

**Answer:**

Stale indexes, missing ACL filters, bad table chunking, over-long contexts that dilute
attention, and duplicated/conflicting sources without ranking.

Another is “pretty context” that looks relevant but is not authoritative—reranking and source
trust signals help.

Finally, naive chunking of code or logs often destroys syntax; specialized splitters matter.

### Q10

**Question:** When not use RAG (fine-tune, workflow, tools)?

**Answer:**

When knowledge is small and stable, fine-tuning or memorization in weights can be simpler than
operating a retrieval stack.

When answers must be deterministic from APIs (balances, inventory), call tools/workflows instead
of asking an LLM to recall.

If latency budgets cannot tolerate retrieval hops, you may need aggressive caching or
precomputed answers for top intents.

---

## 7 — Model routing

### Q1

**Question:** What is model routing?

**Answer:**

Model routing chooses which model (size/provider/capability tier) handles a request based on
signals like intent, risk, user plan, latency SLO, or language.

It is a product + reliability tool: not every query needs a frontier model, but some do—and you
need guardrails so routing doesn’t silently degrade quality.

I implement routing as explicit policy with telemetry, not hidden heuristics scattered in code.

### Q2

**Question:** Why route cheap vs expensive models?

**Answer:**

Cost and latency scale superlinearly with the largest models; many tasks—classification,
rewriting, simple extraction—work on smaller models if validated offline.

The business case is margin: support chat at scale often cannot afford GPT-class on every
message.

The risk is quality cliffs: I shadow-test routes and monitor cohort metrics before widening
cheap paths.

### Q3

**Question:** What signals can drive routing?

**Answer:**

Intent classifiers, estimated difficulty from heuristics (length, presence of code), user tier,
max output length, language, prior failures, and safety categories.

Some teams use a tiny router model or embeddings over historical labeled traffic.

Signals should be explainable in audits—opaque routing is hard to defend in regulated domains.

### Q4

**Question:** How avoid routing being flaky?

**Answer:**

Version routing rules, log decisions with inputs, use feature flags, and roll out gradually with
instant rollback.

Add canaries that compare outputs between routes on sampled traffic when feasible.

Flaky routing often comes from prompt drift in the router—treat router prompts like production
code.

### Q5

**Question:** How does routing interact with caching?

**Answer:**

Cache keys must include model id and prompt version; otherwise you serve a small-model answer
for a query that required a larger model.

Semantic caches likewise need to encode capability tier—similar text is not always similar risk.

I document cache invalidation whenever routing policies change.

### Q6

**Question:** What is shadow routing?

**Answer:**

Shadow routing runs a candidate model off the user-critical path and compares outputs, latency,
and cost against production—useful before promoting a cheaper route.

It reduces risk: you learn failure modes without exposing users.

You still need privacy controls—shadow calls can leak sensitive prompts if misconfigured.

### Q7

**Question:** How measure routing success?

**Answer:**

Compare task completion, error rates, human thumbs-down, escalations, and revenue proxies
between cohorts with similar traffic splits.

Watch for Simpson’s paradox: a route can look good overall but harm a segment.

Statistical rigor matters for small deltas—don’t declare victory on day one noise.

### Q8

**Question:** What are safety concerns with routing?

**Answer:**

Sensitive workflows must never accidentally downgrade to under-monitored endpoints; policy
should be explicit per tenant and data class.

Routing can create inconsistent policy enforcement if different models interpret safety
instructions differently.

Audit logs should include route choice and model version for incident response.

### Q9

**Question:** How implement routing in config vs code?

**Answer:**

I prefer config/feature flags for thresholds and model maps so PMs and on-call can adjust
quickly, while keeping execution paths thin and unit-tested.

Complex conditional logic still belongs in typed code with tests—not YAML spaghetti.

Every route path should have integration tests for at least one representative query.

### Q10

**Question:** GraphChainSQL tie-in for routing?

**Answer:**

In this repo, chat and embedding model names typically come from `Settings`/environment
variables, so you can swap models per environment without code edits—useful for dev vs prod and
cost experiments.

I’d still add explicit experiment tags in traces when comparing models so analysis is clean.

Routing is not only LLM choice: SQL execution path vs pure NL path is another form of routing.

---

## 8 — Streaming

### Q1

**Question:** What is streaming in LLM apps?

**Answer:**

Streaming progressively sends model output—tokens—or workflow events to the client as they are
produced, instead of buffering the full response.

It improves perceived responsiveness and enables richer UIs (step logs, partial renders).

It complicates parsing, retries, and infra buffering—needs deliberate design.

### Q2

**Question:** SSE vs WebSockets?

**Answer:**

SSE is one-way server→client over HTTP, simple to operate through many proxies and great for
token streams and progress logs.

WebSockets are bidirectional when the client must stream audio/video or send frequent control
messages with low overhead.

I default to SSE for LLM token streaming unless I need true bidirectional channels.

### Q3

**Question:** How does streaming improve perceived latency?

**Answer:**

Time-to-first-token drops psychologically; users tolerate waits better when progress is visible.

Even if total wall time is similar, the experience feels faster and more trustworthy.

Product teams often care as much about TTFT as total latency for chat UX.

### Q4

**Question:** What breaks SSE in production?

**Answer:**

Proxies buffering responses, missing `text/event-stream` headers, gzip middleware delaying
flush, aggressive timeouts on load balancers, and corporate middleboxes.

Fixes include disabling buffering on nginx/Cloudflare rules, heartbeat comments, and correct
cache-control.

I validate streaming in staging with the same proxy chain as prod—local dev lies.

### Q5

**Question:** How stream agent steps vs tokens?

**Answer:**

Token streaming is fine-grained LLM deltas; step streaming emits higher-level events like
“retrieved docs,” “called tool X,” “validator failed.”

UIs for agents often need step streaming for transparency; token streaming alone obscures tool
failures.

I define a stable event schema and version it to avoid breaking clients.

### Q6

**Question:** How back-pressure streaming consumers?

**Answer:**

Use bounded queues, pause upstream generation when the client is slow, or degrade to summaries
instead of full token firehoses.

Without back-pressure you buffer unbounded memory server-side or disconnect messily.

Mobile clients especially need resilient reconnect strategies with last-event ids.

### Q7

**Question:** How test streaming endpoints?

**Answer:**

Integration tests that read the stream incrementally, assert ordering, schema of events, and
terminal markers; contract tests for client parsers.

Flaky tests often come from timeouts—tune waits and use deterministic mocks for model streams.

I also test cancellation: when the user aborts, upstream work should stop to save cost.

### Q8

**Question:** How correlate streamed events with traces?

**Answer:**

Emit trace/span ids or a session id in the first event and propagate through subsequent events
so LangSmith/OTEL timelines line up with the UI.

That makes support tickets debuggable: from a screenshot to a trace.

Correlation ids should not include secrets—treat them like JWT jtis.

### Q9

**Question:** When avoid streaming?

**Answer:**

Batch ETL, strict atomic JSON APIs, audit pipelines needing full payloads, or clients that
cannot parse partial data safely.

Some compliance reviews prefer non-streaming logs of final outputs only—policy matters.

If streaming adds complexity without UX benefit, skip it.

### Q10

**Question:** GraphChainSQL tie-in?

**Answer:**

`POST /api/query/stream` streams LangGraph updates so the UI can render graph progress—pair that
with server-side tracing on the same run id.

I’d document the event types consumers should rely on versus internal debug fields.

Streaming plus SQL execution means careful error framing—don’t leak DB internals in SSE
payloads.

---

## 9 — AI agents

### Q1

**Question:** What is an AI agent vs a basic LLM app?

**Answer:**

A basic LLM app is often one or a few model calls with fixed steps; an agent adds a loop where
the model selects actions (tools) based on observations until a goal is reached.

Agents trade flexibility for control: more paths through the system means more testing and
monitoring burden.

I reserve the term “agent” for systems with tool side effects and multi-step autonomy—not every
chatbot.

### Q2

**Question:** Core components of an agent architecture?

**Answer:**

Policy (LLM), tool interfaces, memory, planner/router, guardrails (policy/safety), and
observability—each should be swappable and testable in isolation where possible.

Execution environment matters: sandboxing, credentials, and rate limits are part of the
architecture.

Human oversight hooks are a component too, not an afterthought.

### Q3

**Question:** How reasoning, planning, memory, tools work together?

**Answer:**

Memory supplies relevant context; planning chooses the next step; tools execute real-world
actions and return observations; reasoning integrates observations into the next decision.

In good designs, tools return structured facts, not huge blobs of text, to keep the loop
efficient.

Bad designs dump entire logs into the prompt each iteration—cost explodes and quality drops.

### Q4

**Question:** When choose an agent vs deterministic workflow?

**Answer:**

Use agents when the solution path varies widely with intermediate evidence and tool use is
essential.

Use workflows when compliance requires predictable steps, approvals, and reproducible audits.

Hybrid is common: workflow skeleton with small agent subgraphs inside nodes.

### Q5

**Question:** Prevent infinite loops?

**Answer:**

Hard caps on iterations and tool calls, stop conditions based on structured success checks,
watchdog timers, and circuit breakers on repeated failures.

Also detect semantic loops: the model repeating the same ineffective action—sometimes a
secondary checker helps.

Product UX should surface “I’m stuck” gracefully instead of spinning forever.

### Q6

**Question:** Design safe tool use?

**Answer:**

Least privilege credentials, input validation, idempotent tools, human approval for destructive
ops, and dry-run modes where possible.

Tools should return explicit error codes and safe messages—never raw stack traces to the model
or user.

I treat tool schemas like public APIs: backward compatibility and deprecation matter.

### Q7

**Question:** Evaluate agent decisions, not only fluency?

**Answer:**

Task success metrics, tool correctness, constraint violations, side-effect checks (in mocked
envs), and human rubrics on traces.

Offline datasets of multi-step tasks help; online monitoring catches drift.

Agents need eval harnesses that assert on state transitions, not only final strings.

### Q8

**Question:** Key observability signals for agents?

**Answer:**

Per-step latency, tool error rates, loop counts, token usage per step, retrieval quality, and
route decisions in multi-agent setups.

I dashboard cost per successful task—agents can hide runaway spend behind plausible text.

Alerting on anomalous tool fan-out catches abuse and bugs early.

### Q9

**Question:** Add memory without context bloat?

**Answer:**

Summaries with anchors to source ids, vector retrieval over past sessions, structured state
instead of raw chat logs, and strict token budgets.

I prefer explicit memory writes (“remember this fact”) over implicit endless logging.

Retention policies must align with privacy requirements.

### Q10

**Question:** Risks deploying autonomous agents?

**Answer:**

Runaway spend, data exfiltration via tools, incorrect side effects, prompt injection steering
tool use, and opaque failures that erode trust.

Mitigations include budgets, allowlists, egress controls, approvals, tracing, and gradual
autonomy with kill switches.

At 5 YOE I emphasize organizational readiness: agents need owners, SLOs, and incident playbooks.

---

## 10 — ReAct

### Q1

**Question:** What is the ReAct pattern?

**Answer:**

ReAct interleaves natural-language reasoning with tool actions: think briefly, choose a tool,
observe the result, repeat.

It improves interpretability versus opaque multi-tool calls because the scratchpad explains
intent—though scratchpads cost tokens.

It is a pattern, not a guarantee of correctness: bad reasoning still happens.

### Q2

**Question:** How does ReAct combine reasoning and acting?

**Answer:**

The model emits a rationale (often monitored for quality) and a structured tool invocation based
on that rationale; observations feed back into the next rationale.

This grounds decisions in external facts rather than pure parametric knowledge.

Production systems often constrain the scratchpad format to make parsing and auditing reliable.

### Q3

**Question:** Advantages over single-shot prompting?

**Answer:**

Multi-step tasks become feasible: search, read, compute, verify—single-shot often guesses.

Intermediate observations reduce hallucinated facts when tools are trustworthy.

Trade-offs are latency and fragility: more steps mean more chances for derailment.

### Q4

**Question:** Good tasks for ReAct?

**Answer:**

Research with search/APIs, ops runbooks with verification, ticket triage with lookups—tasks
where external data changes the answer.

Poor fit: simple FAQs where a chain is cheaper and safer.

I match tool granularity to the task—too many micro-tools add decision noise.

### Q5

**Question:** Common failure modes?

**Answer:**

Wrong tool selection, hallucinated tool arguments, loops, overly long scratchpads, and
misinterpreting tool errors.

Another failure is trusting bad tools: garbage observations poison reasoning.

Mitigation includes argument validators, repair prompts, and max-step limits.

### Q6

**Question:** Constrain tool use?

**Answer:**

Small allowlisted tool sets, JSON schema validation, forced disambiguation questions, and
confidence thresholds that route to humans.

Sometimes a planner proposes a tool plan and a cheaper model executes—separation of concerns.

Policy prompts alone are insufficient; enforce constraints in code.

### Q7

**Question:** Evaluate chosen actions?

**Answer:**

Gold tool traces on benchmarks, state-machine assertions in tests, and counterfactual replays
with mocked tools to ensure arguments match expected shapes.

Online, I slice failures by tool name and error code to prioritize fixes.

Human review on sampled traces remains valuable for nuanced tasks.

### Q8

**Question:** Replace ReAct with plan-and-execute or graphs?

**Answer:**

When global planning, parallelization, or strict HITL gates matter, explicit graphs or plan-
execute patterns reduce chaos versus a free-form loop.

Plan-and-execute can reduce tool thrash by committing to a plan first—then execute with fewer
LLM calls.

The right choice depends on task structure and risk tolerance, not trendiness.

### Q9

**Question:** ReAct impact on latency and tokens?

**Answer:**

Each loop adds model round trips and growing prompts if you append full histories naively.

Mitigations: summarize observations, store structured state, cache stable subqueries, and
parallelize independent tool calls when safe.

I set budgets per user/session to prevent death spirals.

### Q10

**Question:** Debug wrong tool picks?

**Answer:**

Log the exact tool schemas the model saw, compare user intent embeddings to tool descriptions,
add targeted few-shot repairs, and consider a small verifier model for high-risk tools.

Often the issue is ambiguous tool names or overlapping responsibilities between tools.

A/B test prompt and schema tweaks with trace-backed metrics, not intuition alone.

---

## 11 — Supervisor agent

### Q1

**Question:** What is a supervisor in multi-agent systems?

**Answer:**

A supervisor is a coordinator—either a dedicated model/node or policy layer—that routes subtasks
to specialist workers and merges their outputs toward a global goal.

It centralizes control flow, which simplifies debugging compared to fully peer-to-peer agent
chatter.

The risk is a bottleneck: the supervisor can become a single point of latency and error
amplification.

### Q2

**Question:** Supervisor vs worker responsibilities?

**Answer:**

The supervisor decides who should act next and what success looks like at a high level; workers
execute domain-specific steps with narrow tools and prompts.

Workers should not silently change global goals; the supervisor owns prioritization and
escalation.

Clear interfaces prevent workers from re-litigating routing decisions ad hoc.

### Q3

**Question:** How supervisor picks an agent?

**Answer:**

Options include rules (regex/intent), a lightweight classifier, an LLM router with schema-
constrained outputs, or embeddings over task descriptions matched to agent capability cards.

Whatever you pick, document the policy and test it—routing bugs look like “random specialist.”

I log routing confidence and alternatives considered when using LLM routers for postmortems.

### Q4

**Question:** Design message passing?

**Answer:**

Prefer structured state keys (task spec, artifacts, errors) over dumping entire chat histories
to each worker.

Version message schemas so you can evolve fields without breaking replay of old traces.

Include correlation ids on every handoff for distributed tracing.

### Q5

**Question:** What state should a supervisor keep?

**Answer:**

Goal decomposition, per-subtask status, partial artifacts, consolidated citations, error
aggregates, and human decision flags.

Enough to resume after crashes: think durable workflow state, not ephemeral prompt text.

Avoid storing secrets in plaintext state; reference secure handles instead.

### Q6

**Question:** Prevent supervisor bottleneck?

**Answer:**

Parallelize independent subtasks, keep supervisor prompts compact, cache stable routing
decisions, and move heavy reasoning into workers.

Sometimes a two-tier supervisor helps: fast rule router first, LLM supervisor only on ambiguous
cases.

Monitor supervisor p95 separately from worker p95—tail latency hides there.

### Q7

**Question:** When retry, reroute, escalate, terminate?

**Answer:**

Retry transient tool/network errors with backoff; reroute when capability mismatch or repeated
failure suggests a different specialist.

Escalate to humans on policy risk, low confidence, or sensitive side effects; terminate on
success or hard failure budgets.

Make these policies explicit in code/graph edges—not only implied by prompts.

### Q8

**Question:** Trace failures in supervisor-worker setups?

**Answer:**

Propagate trace ids across handoffs, log each worker’s inputs/outputs with redaction, and
persist enough state to replay a failed run.

For multi-agent incidents, the sequence graph is the narrative—logs should mirror it.

I add synthetic probes that exercise each routing edge nightly.

### Q9

**Question:** Human approval in supervisor-led systems?

**Answer:**

The supervisor sets flags like `needs_approval` and pauses orchestration until an authenticated
human API resumes with allow/deny, writing the decision into state.

Approvals should be enforced by the workflow engine, not “ask nicely” in model text.

Audit trails include who approved, when, and what evidence they saw.

### Q10

**Question:** Centralized vs decentralized collaboration trade-offs?

**Answer:**

Centralized supervisors are easier to reason about and secure; decentralized peer agents can
reduce single-point bottlenecks but complicate policy consistency.

Org maturity matters: decentralized setups need strong contracts and observability discipline.

I often start centralized, then decentralize only where measured need proves it.

---

## 12 — Multi-agent systems

### Q1

**Question:** What is a multi-agent system, and why use it?

**Answer:**

Multiple specialized agents collaborate—retrieval, coding, verification—often coordinated by a
router or supervisor.

Use it when task decomposition genuinely reduces error or leverages different tools/models per
subtask.

Don’t use it for resume theater: extra agents add latency, coordination bugs, and debugging
surface.

### Q2

**Question:** Common roles split across agents?

**Answer:**

Retriever, analyst, coder, tool executor, critic/verifier, summarizer—each with narrow prompts
and least-privilege tools.

The critic pattern catches subtle mistakes before user-facing output.

Role explosion is a smell: merge agents until metrics justify the split.

### Q3

**Question:** Trade-offs specialization vs coordination?

**Answer:**

Specialization improves expertise per step but increases handoff overhead and the chance of
inconsistent assumptions.

Coordination costs show up in tokens, latency, and failure rates—measure end-to-end task
success, not per-agent fluency.

I prefer fewer, better-instrumented agents over many shallow ones.

### Q4

**Question:** Share context safely?

**Answer:**

Pass summaries plus references to underlying docs with ACL checks; never forward full corpora
across agents blindly.

Enforce tenant isolation at retrieval time, not only at the UI.

Secrets should never traverse agent prompts—use short-lived tokens scoped to the tool layer.

### Q5

**Question:** Prevent contradictory outputs?

**Answer:**

Add a critic/consensus step, maintain a single source of truth field in state for final answers,
or require structured agreement between agents.

Without merge rules, users see inconsistent narratives across turns.

Tests should include adversarial disagreements between mocked agents.

### Q6

**Question:** Sequential vs parallel vs hierarchical workflows?

**Answer:**

Sequential is simplest to debug; parallel speeds independent evidence gathering; hierarchical
mirrors org processes (manager/worker) and can map to approvals.

Parallel requires reducers/merge semantics and idempotent tools.

Pick structure based on dependency graph of the task, not diagram aesthetics.

### Q7

**Question:** When is multi-agent unnecessary?

**Answer:**

When a single RAG chain or small tool loop meets quality and SLOs—additional agents rarely help.

If your bottleneck is data quality, agents won’t fix it.

Start minimal, add agents only when traces show a clear, repeatable decomposition win.

### Q8

**Question:** Benchmark multi-agent vs single baseline?

**Answer:**

Same golden tasks, measure success rate, cost, latency, human rework, and safety violations
across variants.

Segment by difficulty: multi-agent sometimes helps only on hard tail queries.

Watch for regressions on easy queries—complex systems often hurt the median case.

### Q9

**Question:** Isolate failures so one weak agent does not corrupt all?

**Answer:**

Per-node try/except with structured errors, circuit breakers per tool provider, and fallbacks to
simpler paths (e.g., direct retrieval answer without research agent).

Never let one worker’s malformed output overwrite authoritative state without validation.

Chaos test individual worker outages.

### Q10

**Question:** Implement orchestration in LangGraph?

**Answer:**

Model specialists as nodes/subgraphs, route explicitly, checkpoint shared state, and trace each
node—this repo’s pipeline pattern is a good anchor for discussion.

Keep subgraph contracts narrow: inputs/outputs documented and tested.

Use thread isolation and idempotent side effects for safe replays.

---

## 13 — Scenario-based questions

### Q1

**Question:** Design customer support FAQ + escalation agent.

**Answer:**

Tier-1: fast RAG over help articles with citations; include confidence scoring from retrieval
density, reranker margin, or a verifier for policy questions.

Escalation path: bundle the trace, user tier, attempted answers, and retrieved doc ids for the
human agent—minimize repeated customer effort.

Guardrails: PII redaction, tool restrictions on account changes, and rate limits. Measure
deflection vs CSAT trade-offs continuously.

### Q2

**Question:** Design research agent: search, validate sources, report.

**Answer:**

Use tools to fetch multiple sources, extract publication dates, cross-check claims, and store
structured notes in state for auditability.

Add a critic step that flags low corroboration or stale sources before final synthesis.

Cap tool spend and steps; research agents love infinite browsing if unconstrained.

### Q3

**Question:** Legal assistant hallucinates despite good retrieval—debug?

**Answer:**

First verify the claim exists verbatim in retrieved chunks—often the model paraphrases beyond
evidence.

Tighten instructions to answer only from cited spans, reduce context noise, add a verifier pass,
and measure faithfulness on a legal golden set.

Check for authoritative vs secondary sources ranking—wrong chunk can be “close” semantically but
not legally binding.

### Q4

**Question:** RAG works in test but fails on long enterprise docs—first change?

**Answer:**

Improve structure-aware chunking and hierarchical retrieval: summarize sections, retrieve
summaries first, then drill into leaf chunks.

Increase relevant retrieval budget carefully and add reranking tuned to long documents.

Often the issue is tables/appendices being mangled—specialized parsers beat naive text splits.

### Q5

**Question:** Accurate but slow—reduce latency without huge quality loss?

**Answer:**

Cache repeated queries, route easy intents to smaller models, parallelize retrieval, trim
context, stream partial results, and skip rerank when confidence is high.

Measure each optimization against a frozen golden set so you don’t silently trade quality.

Regional placement and connection pooling fix a surprising amount of “model slowness.”

### Q6

**Question:** Relevant chunks but final answer misses facts—diagnosis?

**Answer:**

Likely context overload/dilution, wrong chunk ordering, or instruction-following failure—reduce
chunks, rerank, or add an intermediate extraction step that pulls bullet facts before
generation.

Sometimes the model optimizes for brevity and drops constraints—tune instructions and penalties
for omissions.

Inspect attention-heavy failure cases with shorter prompts to isolate the issue.

### Q7

**Question:** Agent double-books meetings from duplicate tool calls—fix?

**Answer:**

Make booking tools idempotent with client request ids, add server-side dedup, use transactional
calendar APIs, and detect duplicate tool calls within a short window.

At orchestration level, disallow repeated writes without monotonic state transitions.

Post-incident, add tests that replay parallel tool calls.

### Q8

**Question:** 100k daily users, control token spend—architecture?

**Answer:**

Semantic and exact caches, per-user/org quotas, model routing, aggressive summarization of
history, batch offline workloads, and anomaly alerts on token spikes.

Separate synchronous chat path from async heavy analysis to protect p95.

Finance should see cost per successful resolution—not only raw token totals.

### Q9

**Question:** Multi-agent inconsistent answers across runs—improve determinism?

**Answer:**

Lower temperature on critical steps, freeze prompts and schemas, record seed where applicable,
add self-consistency checks for high-risk outputs, and trace all randomness sources.

Reduce the number of free-form natural language decisions—push more into structured routing.

Sometimes inconsistency is data drift; verify corpus version pinned in traces.

### Q10

**Question:** Context contains answer but model cites wrong chunk—improve grounding?

**Answer:**

Force citation indices tied to provided chunks, retrieve fewer higher-precision chunks,
train/evaluate with citation-aware metrics, and reorder by reranker margin.

Add a post-check that cited chunk text supports each sentence.

If citations are cosmetic, models learn to game them—make citations load-bearing in evaluation.

---

## 14 — Latency

### Q1

**Question:** Main latency sources in LLM apps?

**Answer:**

Network RTT, queueing, model time-to-first-token and decode speed, retrieval, reranking, tool
calls, serialization, and client rendering.

People underestimate tool RTT: a slow internal API dominates LLM time.

Profiling should be end-to-end, not only the model vendor dashboard.

### Q2

**Question:** How retrieval, reranking, tools affect E2E latency?

**Answer:**

They usually add sequential hops unless explicitly parallelized; each hop adds tail risk.

Fan-out multiplies worst-case latency unless you cap concurrency and use timeouts judiciously.

I build a latency budget table per product SLO and assign caps to each hop.

### Q3

**Question:** Techniques to reduce RAG latency?

**Answer:**

Tune ANN parameters, cache query embeddings, reduce chunk count, regional indexes, prefetch for
known flows, and skip rerank on easy queries.

Warm connections and reuse HTTP clients—milliseconds matter at scale.

Sometimes smaller embedding models plus rerank beat huge embeddings alone on latency.

### Q4

**Question:** Trade quality for latency when?

**Answer:**

When product SLOs or regulatory time windows require it— but document the trade and monitor
quality cohorts so regressions are detectable.

Use feature flags to roll back if the faster path hurts conversion or escalations.

Prefer principled degradation (shorter context) over silent model downgrades without
measurement.

### Q5

**Question:** How streaming improves perceived latency?

**Answer:**

Users see tokens or steps sooner, which improves satisfaction even when total time is unchanged.

It also helps diagnose stuck workflows earlier in the UI.

Pair streaming with heartbeats so users know the connection is alive.

### Q6

**Question:** Model size and context length vs response time?

**Answer:**

Larger models generally increase prefill and decode cost; longer contexts increase prefill
substantially.

I measure tokens/sec and TTFT separately—optimizations differ for each.

Sometimes trimming 2k tokens of junk context beats swapping to a bigger model.

### Q7

**Question:** Profile latency across agent nodes?

**Answer:**

Distributed tracing (OTEL/LangSmith) with spans per node/tool and server timers around each
critical section.

Aggregate by graph version to catch regressions when prompts change.

Include queue wait time, not only execution time, in spans.

### Q8

**Question:** Reranking latency impact and control?

**Answer:**

Reranking adds a cross-encoder or LLM judge step—control candidate count, model size, and
batching.

Offer async rerank for non-interactive surfaces or premium tiers.

Cache rerank results for stable query-chunk pairs when privacy allows.

### Q9

**Question:** Concurrency, batching, caching at scale?

**Answer:**

Async I/O, bounded thread pools, batch embedding where safe, multi-layer caches with TTLs, and
coalescing duplicate in-flight requests.

Watch cache stampede on hot keys—singleflight patterns help.

Concurrency increases tail latency unless you cap and shed load.

### Q10

**Question:** Which latency metrics in prod (p50, p95, p99)?

**Answer:**

All of them: p50 for typical UX, p95/p99 for SLAs and incident response—averages hide unhappy
users.

Segment by region, tenant size, and intent class—aggregate metrics lie.

Pair latency with error rate and cost to avoid “fast but wrong.”

---

## 15 — Cost and tokens

### Q1

**Question:** What is a token, and why it matters?

**Answer:**

Tokens are the billing units for most LLM APIs; both prompt and completion tokens drive cost and
often latency (especially prompt tokens for prefill).

Tokenizer quirks mean dollars don’t line up with “words” intuitively—finance needs token-based
models.

Engineering decisions (context stuffing, history logging) directly hit margin.

### Q2

**Question:** How prompt size and retrieved context affect cost?

**Answer:**

You pay for every prompt token on every call; large RAG contexts multiply cost across user
sessions and agent loops.

Agents re-send history each step if you’re not careful—token spend explodes.

I track tokens per successful task, not per call, to guide architecture.

### Q3

**Question:** Strategies to control API spend?

**Answer:**

Caching (exact/semantic), routing to smaller models, summarizing history, reducing tool chatter,
quotas, budget alerts, and batching offline workloads.

Kill switches for runaway agents and per-org spend caps prevent surprises.

Product-led limits (max attachments, max analysis minutes) also help.

### Q4

**Question:** How caching and compression reduce tokens?

**Answer:**

Avoid recomputing identical or near-identical prompts; compress older turns into summaries while
preserving key facts and citations.

Be careful: semantic caches can serve wrong answers if matching is too fuzzy—tie with confidence
thresholds.

Invalidate caches on prompt/model version changes.

### Q5

**Question:** When route to smaller/cheaper models?

**Answer:**

For high-volume, low-risk tasks like classification or formatting—after offline validation shows
parity on key metrics.

Roll out gradually with shadow traffic and online monitoring.

Never route safety-critical flows without explicit policy gates.

### Q6

**Question:** Token quotas and budget alerts?

**Answer:**

Per-user/org counters in Redis/DB, soft warnings, hard blocks, anomaly detection on spikes, and
finance dashboards tied to product metrics.

Alerts should include likely root causes: new feature flag, cache miss spike, or prompt
regression.

Quotas need UX that explains limits clearly to users.

### Q7

**Question:** Why multi-step agents cost more than chains?

**Answer:**

Each loop repeats prompt prefixes and accumulates observations; without state discipline you pay
multiple times for the same content.

Mitigate with structured state, summaries, and caps on iterations/tool calls.

Compare agent cost to human equivalent work to keep perspective—but still control tail spend.

### Q8

**Question:** Estimate monthly chatbot cost at high traffic?

**Answer:**

Build a spreadsheet model: average prompt/completion tokens × price × requests, plus retrieval,
reranking, and infra. Calibrate with a week of sampled production logs.

Include tail traffic and agent loops—medians underestimate.

Revisit after major prompt changes; costs shift silently.

### Q9

**Question:** Trade-offs larger context vs retrieval?

**Answer:**

Huge contexts reduce retrieval engineering but increase cost and can confuse models with noise.

Often retrieving fewer, sharper chunks beats stuffing everything “because we can.”

Measure faithfulness and factuality as context grows—more isn’t always better.

### Q10

**Question:** Cost metrics per request/user/workflow?

**Answer:**

LLM dollars per successful task, retrieval/rerank dollars, infra amortized, and downstream
support rework cost.

Tie technical metrics to business KPIs so cost cuts don’t tank revenue.

Attribute multi-step workflows with trace-level accounting.

---

## 16 — Failure handling

### Q1

**Question:** Common failure modes in LLM/agent systems?

**Answer:**

Tool timeouts, malformed JSON arguments, retrieval misses, policy violations, rate limits, model
outages, and silent hallucinations where outputs look fluent but wrong.

Distributed systems add partial failures: one dependency degrades while others succeed,
producing inconsistent state if not handled.

I classify failures by layer—retrieval, tool, model, orchestration—to avoid generic “AI broke”
tickets.

### Q2

**Question:** Distinguish retrieval vs generation failure in RAG?

**Answer:**

If labeled gold evidence is not in top-k, that is retrieval—fix chunking, embeddings, filters,
hybrid, rerank.

If gold is present but ignored or misused, that is generation/instruction following—reduce
noise, tighten prompts, reorder context, or add extraction.

Automate checks with trace-linked chunk ids so triage is fast during incidents.

### Q3

**Question:** Tool timeout or malformed data—how agent responds?

**Answer:**

Return structured errors to the model with safe summaries, allow bounded retries with backoff,
and if unrecoverable, produce a user-facing message that preserves trust without leaking
internals.

Never infinite-retry writes; distinguish read vs write retry semantics.

Log correlation ids and tool latency percentiles to catch systemic outages vs single-row bad
data.

### Q4

**Question:** Retries without duplicate actions?

**Answer:**

Use idempotent APIs with idempotency keys, server-side dedup, and transactional semantics for
bookings/payments.

For reads, retries are safer; for writes, require explicit confirmation or monotonic state
transitions.

Tests should cover duplicate delivery scenarios explicitly.

### Q5

**Question:** Prevent confident-but-wrong silent failures?

**Answer:**

Uncertainty thresholds, abstain behaviors, second-pass verification, mandatory citations for
factual claims, and user feedback capture on bad answers.

Online monitors on citation validity and tool error spikes catch drift early.

Culture matters: reward reporting uncertainty, not only “helpful” answers.

### Q6

**Question:** Fallback if LLM fails mid-task?

**Answer:**

Serve a degraded answer path (cached FAQ), switch to a simpler model, or hand off to humans with
full state and trace attached.

Never leave users hanging without messaging; preserve partial progress when safe.

Feature flags should allow instant disable of risky subgraphs.

### Q7

**Question:** Guardrails for payments/bookings/sensitive data?

**Answer:**

Step-up authentication, explicit user confirmations, amount limits, immutable audit logs, and
separation of duties between suggestion and execution.

Tools executing money movement should be non-LLM-controlled finalizers when possible.

Regulatory contexts need retention and access controls on traces containing PII.

### Q8

**Question:** Essential logging/tracing for prod failures?

**Answer:**

Trace ids, prompt versions, model ids, retrieval chunk ids, tool payloads redacted, per-node
timings, and final routing decisions.

Enough to replay the decision path in staging with mocks.

Balance verbosity with PII risk—log references, not full documents, where possible.

### Q9

**Question:** Recover when one agent fails in multi-agent flow?

**Answer:**

Circuit-break the failing branch, fall back to a simpler path, or escalate with partial
results—never fail closed without user-visible explanation.

Supervisor should merge partial successes when policy allows.

Postmortems should update tests for the missing edge case.

### Q10

**Question:** Test failure scenarios pre-deploy?

**Answer:**

Chaos tests for dependencies, contract tests for tools, fuzzed arguments, and replay tests from
recorded traces.

Game days for rate limits and model outages build muscle memory.

Automate detection of new unhandled exception types in staging logs.

---

## 17 — Testing (unit, integration, CI)

### Q1

**Question:** What should you unit test in LLM apps?

**Answer:**

Deterministic pieces: parsers, validators, routing rules, token accounting, prompt template
rendering with fixtures, and pure transforms of retrieved text.

Avoid asserting exact model prose except in pinned golden tests with known stochasticity
controls.

Unit tests are the fast safety net on every PR.

### Q2

**Question:** What should integration tests cover?

**Answer:**

API contracts, auth, database migrations, retrieval pipelines with mocked embeddings/vector
responses, and tool mocks returning edge cases (timeouts, partial JSON).

Include streaming contracts if you ship SSE/WebSockets.

Integration tests catch wiring mistakes unit tests miss.

### Q3

**Question:** How test prompts safely?

**Answer:**

Snapshot template strings minus secrets, table-driven variable injection tests, and schema
checks for structured outputs.

Golden outputs only when the model/version/temperature is pinned and tolerances defined.

Track prompt hashes in CI to detect accidental edits.

### Q4

**Question:** How CI fits LLM projects?

**Answer:**

Fast lint/unit on every PR; heavier eval suites nightly or on-demand; block merges on
deterministic failures only to keep velocity.

Separate “smoke eval” (minutes) from “full research eval” (hours).

CI should never depend on flaky external APIs without mocks.

### Q5

**Question:** How mock LLM calls in tests?

**Answer:**

Dependency injection of the client, httpx mock transports, or provider stub servers;
record/replay when allowed by policy.

Fixtures should include refusals and malformed tool calls, not only happy paths.

Keep mocks aligned with real schema drift via contract tests.

### Q6

**Question:** What are contract tests for tools?

**Answer:**

Assert JSON schemas, error codes, idempotency behavior, pagination, and auth failures against a
stub server that mirrors production semantics.

They prevent agents from silently breaking when backend teams ship changes.

Version the contract when tools evolve.

### Q7

**Question:** How test streaming endpoints?

**Answer:**

Read streams incrementally, assert ordering, schema per event, terminal events, and cancellation
behavior.

Include timeout and reconnect simulations for mobile clients.

Snapshot only stable fields—avoid timestamps in strict asserts.

### Q8

**Question:** How prevent flaky LLM tests?

**Answer:**

Pin model+version+temperature, assert structured fields, use statistical tolerances for judges,
and rerun policies sparingly.

Flaky tests erode trust—fix root causes, not thresholds blindly.

Prefer deterministic validators over LLM judges in CI when possible.

### Q9

**Question:** What is smoke testing after deploy?

**Answer:**

Minimal live checks: health endpoints, one canonical RAG query, maybe a streaming sanity
check—enough to validate config, connectivity, and critical paths.

Run from the same network path users hit (DNS, CDN, proxies).

Alert immediately on failure with rollback hooks ready.

### Q10

**Question:** GraphChainSQL tie-in?

**Answer:**

`test_graph.py`, `test_query.py`, and API route tests benefit from deterministic configuration,
mocked externals, and explicit fixtures for graph transitions.

When touching agents, add tests around state updates and error propagation—not only HTTP 200.

Streaming routes need incremental readers in tests mirroring client behavior.

---

## 18 — Offline testing & model selection

### Q1

**Question:** What is offline testing for LLM systems?

**Answer:**

Running the full pipeline on frozen datasets without live user traffic—used for regression
detection, model comparisons, and prompt iteration with reproducibility.

It complements online experiments: cheaper, safer, and easier to debug.

Offline success is necessary but not sufficient—distribution shift always remains.

### Q2

**Question:** Why offline before online A/B?

**Answer:**

You catch catastrophic failures and obvious regressions cheaply; you protect users and brand
risk.

It also builds statistical priors for what effect sizes are plausible before designing live
tests.

Stakeholders trust offline dashboards when they include slice analysis.

### Q3

**Question:** What datasets do you need?

**Answer:**

Representative queries with labels: expected tools, expected SQL, rubric scores, or reference
answers; plus negative/adversarial cases and long-tail samples from production logs
(appropriately scrubbed).

Balance easy and hard rows—optimizing only on hard sets can hurt median UX.

Version datasets like code: hash and changelog.

### Q4

**Question:** How compare two models offline?

**Answer:**

Same prompts, same retrieval index version, measure task success, latency, cost, safety
violations, and statistical significance across slices.

Use paired comparisons per query when possible for sensitivity.

Watch for judge bias if using automated evaluators—calibrate against humans.

### Q5

**Question:** How detect regressions when prompts change?

**Answer:**

Run golden suites on each PR with thresholds; alert on metric drops beyond noise; require human
review for large prompt diffs.

Track which metric regressed (faithfulness vs fluency) to guide fixes.

Pair with qualitative spot checks on worst outliers.

### Q6

**Question:** Role of human eval in offline testing?

**Answer:**

Humans label nuanced cases, adjudicate disagreements between judges, and catch subjective
failures automation misses.

They also calibrate LLM judges over time.

Sampling strategy matters—focus human time on high-risk and high-impact queries.

### Q7

**Question:** How store offline eval results?

**Answer:**

Experiment tables or MLflow-like trackers storing git SHA, dataset hash, model id, prompt
version, metrics, and artifacts.

Reproducibility is the point—store seeds and dependency lock info too.

Retention policies apply if outputs contain sensitive data.

### Q8

**Question:** How long should an offline suite take?

**Answer:**

Minutes for CI-friendly subsets; longer nightly jobs for broad coverage—tuned to team velocity
vs risk.

Parallelize eval workers and cache retrieval results when deterministic.

If suites are too slow, teams skip them—optimize the critical path ruthlessly.

### Q9

**Question:** Offline limitations?

**Answer:**

Train/test mismatch vs production language and intents; stale corpora; missing rare events; and
judge blind spots.

Always pair with online monitoring: escalations, thumbs-down, latency, and safety flags.

Treat offline metrics as guardrails, not proof of user happiness.

### Q10

**Question:** GraphChainSQL tie-in?

**Answer:**

Follow `scripts/offline_eval.py` patterns; optional `RAGAS_COLLECT_ON_COMPLETE` flows into
persisted scores (`src/services/ragas_service.py`) for longitudinal tracking.

Wire eval runs to the same tracing tags you use in prod for apples-to-apples comparisons.

When SQL is involved, include execution correctness and safety (read-only) in offline harnesses.

---

## 19 — A/B testing

### Q1

**Question:** What is A/B testing for LLM features?

**Answer:**

Randomized assignment of users to variants (model, prompt, retrieval strategy) with controlled
traffic and comparable metrics.

It is how you prove incremental lift—not slides claiming a better model.

Requires disciplined logging and ethical review for sensitive domains.

### Q2

**Question:** What metrics for LLM A/B?

**Answer:**

Task success, latency, cost, thumbs-down, escalation rate, revenue proxies, safety incidents—not
BLEU alone.

Pick primary metrics up front to avoid p-hacking across many dashboards.

Segment metrics by cohort (new vs power users) to avoid misleading aggregates.

### Q3

**Question:** How long to run an A/B?

**Answer:**

Until power analysis indicates you can detect the expected effect size with acceptable error
rates—often days to weeks depending on traffic and variance.

LLM metrics can be noisy; premature stops fool teams.

Monitor guardrail metrics daily even if the experiment runs weeks.

### Q4

**Question:** How avoid Simpson's paradox?

**Answer:**

Stratify by geography, device, user tenure, time-of-day, and product surface; validate that
winner is consistent across key slices.

Sometimes a global winner hides harm to a vulnerable segment—ethical and business risk.

Pre-register analysis slices when possible.

### Q5

**Question:** How implement variant assignment?

**Answer:**

Stable hashing on user ids, sticky assignment across sessions, server-side flags—never trust
client-only assignment.

Include experiment id in traces for debugging and for preventing cross-experiment contamination.

Handle logged-out users carefully (cookie/device id stability).

### Q6

**Question:** Ethical considerations?

**Answer:**

Disclose when required by policy/law, avoid harmful cohorts, provide opt-outs where appropriate,
and ensure instant kill switches.

High-stakes domains (mental health, finance) need extra review.

Document data retention for experiment logs.

### Q7

**Question:** How roll out safely?

**Answer:**

Start with 1–5% canaries, watch error budgets and guardrails, ramp gradually, keep instant
rollback and feature flags.

Pair rollout with shadow metrics on the non-canary cohort for sanity.

Communicate cross-functionally: support teams need heads-up on behavior changes.

### Q8

**Question:** How link A/B to tracing?

**Answer:**

Tag traces with `experiment_id` and `variant` so engineers can debug per cohort and slice
failures.

This connects product analytics to root-cause engineering views.

Ensure PII tagging rules still apply in experiment logs.

### Q9

**Question:** When A/B is the wrong tool?

**Answer:**

Tiny traffic can’t reach significance; some experiments cannot be randomized ethically; if
logging is incomplete you can’t measure outcomes.

In those cases use offline evals, qualitative studies, or quasi-experiments with caution.

Don’t A/B test safety policies casually.

### Q10

**Question:** GraphChainSQL tie-in?

**Answer:**

Log `run_id`/session and prompt versions from the database alongside traces so cohort
comparisons line up in LangSmith or OTEL backends.

When comparing SQL-generating variants, include execution success and human correction rate—not
only natural language quality.

Treat schema migrations as incompatible with naive experiment analysis—version everything.

---
