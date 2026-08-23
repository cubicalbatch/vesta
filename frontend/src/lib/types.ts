// Shapes mirror docs/sse-protocol.md exactly. Do not add fields the protocol
// doesn't define — this file is the client-side reading of that doc, not a
// place to invent convenience shapes.

// Mirrors MediaOut in src/vesta/api/zims.py. Present on ArticleOut / search
// SourceCards for media-ZIM entries (0008): the frontend renders a native
// <video poster=… src=…> from it (paths are ZIM-relative, served via /api/zim/).
export interface MediaOut {
	video_path: string | null;
	poster_path: string | null;
	duration: number | null;
}

// Mirrors DocumentOut in src/vesta/api/zims.py (0013). Present on
// GET /api/zims/{id}/documents and on ArticleOut / search SourceCards for
// nautiluszim document-library ZIMs: the manifest title/author/description
// plus the path-preserving reader URL (a PDF served with its true
// application/pdf mimetype via /api/zim/, rendered natively by the browser).
export interface DocumentOut {
	doc_path: string;
	title: string | null;
	description: string | null;
	author: string | null;
	doc_mime: string;
	url: string;
}

export interface SourceCard {
	zim_id: number;
	path: string;
	title: string;
	snippet: string;
	breadcrumb: string;
	score: number;
	source: string;
	/** Client-only: set on cards that arrived via a merge:true sources event. */
	recovered?: boolean;
	/** 0008: set on media-ZIM cards so the Reader/thumbnail can show a video. */
	media?: MediaOut | null;
	/** 0013: set when the card is a document-library entry — manifest title/author live here. */
	document?: DocumentOut | null;
}

export interface CitationSpan {
	answer_span: [number, number];
	card_id: number;
	passage_span: [number, number] | null;
	score: number;
}

export interface TraceStage {
	name: string;
	component: string;
	params: Record<string, unknown>;
	inputs: Record<string, unknown>;
	outputs: Record<string, unknown>;
	duration_ms: number;
}
/** One agent-turn timed step (pre_seed / agent_llm / search / read_article).
 *  Same shape as a retrieval TraceStage, plus optional nested retrieval
 *  pipeline stages for search steps (so encoder/rerank/search timings surface). */
export interface AgentTraceStage extends TraceStage {
	stages?: TraceStage[];
}

/** Retrieval / sources_only trace — versioned, stages-based
 *  (vesta/retrieval/trace.py::Trace.to_dict). */
export interface RetrievalTrace {
	version: number;
	stages: TraceStage[];
	degradations: { component: string; missing: string; reason: string }[];
	profile: string;
	profile_hash?: string;
}

/** pydantic-agent turn trace — a flat summary plus a per-step timing
 *  breakdown emitted by api/agent_chat.py::iter_agent_turn_events.
 *  ``stages`` is the timed step list (pre_seed / agent_llm / search /
 *  read_article), each search step carrying nested retrieval pipeline stages. */
export interface AgentTrace {
	system: string;
	followup?: boolean;
	elapsed_ms: number;
	total_tokens: number;
	input_tokens: number;
	output_tokens: number;
	search_calls: number;
	read_calls: number;
	card_count: number;
	stages?: AgentTraceStage[];
}

/** Any trace shape the answer stream can carry. Discriminate on `'system' in
 *  trace` (AgentTrace) vs `'stages' in trace` (RetrievalTrace). */
export type Trace = RetrievalTrace | AgentTrace;

export type AnswerPhase = 'reading' | 'generating' | 'abstaining' | 'sources_only' | 'searching';

export type AnswerErrorCode =
	| 'no_llm'
	| 'retrieval_failed'
	| 'stream_error'
	| 'fatal'
	| 'no_profile'
	| 'budget_exhausted'
	| 'unknown_event';

export interface AnswerError {
	code: AnswerErrorCode;
	message: string;
	recoverable: boolean;
}

// Discriminated union over the wire's `event:` name, paired with its `data:` payload.
export type AnswerEvent =
	| { event: 'sources'; data: { cards: SourceCard[]; merge?: boolean } }
	| { event: 'status'; data: { phase: AnswerPhase; detail: string } }
	| { event: 'token'; data: { text: string } }
	| { event: 'answer_reset'; data: { reason: string } }
	| { event: 'citations'; data: { spans: CitationSpan[]; answer_text?: string | null } }
	| { event: 'trace'; data: Trace }
	| { event: 'done'; data: Record<string, never> }
	| { event: 'error'; data: AnswerError };

// Job records.
export type JobStatus = 'queued' | 'running' | 'paused' | 'done' | 'error' | 'cancelled';
export type JobType = 'download_zim' | 'index_zim' | 'refresh_catalog' | 'download_model' | 'noop';

export const TERMINAL_JOB_STATUSES: ReadonlySet<JobStatus> = new Set(['done', 'error', 'cancelled']);

export interface JobRecord {
	id: number;
	type: JobType;
	target: string;
	status: JobStatus;
	progress: number | null;
	total: number | null;
	checkpoint: Record<string, unknown> | null;
	params: Record<string, unknown>;
	message: string | null;
	error: string | null;
	rate: number | null;
	eta_seconds: number | null;
	created_at: string;
	updated_at: string;
	finished_at: string | null;
}

export type JobProgressDelta = Pick<JobRecord, 'id' | 'progress' | 'total' | 'message' | 'status'>;

export type JobStreamEvent =
	| { event: 'snapshot'; data: JobRecord }
	| { event: 'progress'; data: JobProgressDelta }
	| { event: 'status'; data: JobRecord };

// Mirrors ArchiveOut in src/vesta/api/zims.py.
export type IndexStatus = 'none' | 'running' | 'paused' | 'complete' | 'stale' | 'error';

export interface Archive {
	id: number;
	uuid: string;
	name: string | null;
	title: string | null;
	language: string | null;
	flavour: string | null;
	file_size: number | null;
	article_count: number;
	has_fulltext_index: boolean;
	corpus_label: string | null;
	kind: 'articles' | 'media' | 'spa' | 'documents';
	scraper: string | null;
	tags: string | null;
	enabled: boolean;
	status: string;
	index_depth: number;
	index_status: IndexStatus;
	embedding_model: string | null;
}

// Mirrors ConversationSummary / MessageDetail / ConversationDetail in
// src/vesta/api/chat.py.
export interface ConversationSummary {
	id: number;
	title: string | null;
	created_at: string | null;
	updated_at: string | null;
}

export interface MessageDetail {
	id: number;
	role: string;
	content: string | null;
	sources: SourceCard[] | null;
	trace: Trace | null;
	tokens_in: number | null;
	tokens_out: number | null;
	latency_ms: number | null;
	created_at: string | null;
}

export interface ConversationDetail {
	conversation: ConversationSummary;
	messages: MessageDetail[];
}

// Mirrors CatalogEntryOut / InstallEstimateOut / CatalogListOut /
// CuratedEntryOut / CatalogStateOut in src/vesta/api/library.py.
export interface InstallEstimate {
	seconds_low: number;
	seconds_high: number;
	vector_bytes: number;
}

export interface CatalogEntry {
	id: string;
	name: string;
	title: string;
	description: string;
	language: string;
	flavour: string;
	tags: string;
	size_bytes: number;
	article_count: number;
	url: string;
	illustration_url: string | null;
	zim_date: string | null;
	curated_rank: number | null;
	curated_warning: string | null;
	fetched_at: string | null;
	install_estimates: Record<string, InstallEstimate>;
}

export interface CatalogList {
	entries: CatalogEntry[];
	total: number;
	available: boolean;
	fetched_at: string | null;
}

export interface CatalogLanguage {
	code: string;
	count: number;
}

export interface CuratedEntry {
	name: string;
	rank: number;
	size_note: string;
	description: string;
	article_count: number;
	warning: string | null;
}

export interface CatalogState {
	count: number;
	fetched_at: string | null;
	available: boolean;
}

export interface IndexEstimate {
	depth: number;
	articles: number;
	seconds_low: number;
	seconds_expected: number;
	seconds_high: number;
	disk_bytes_low: number;
	disk_bytes_expected: number;
	disk_bytes_high: number;
	calibrated: boolean;
}

export interface ScanResult {
	added: number[];
	updated: number[];
	missing: number[];
	total: number;
}

// Mirrors SectionOut / ArticleOut in src/vesta/api/zims.py. Backs both
// GET /api/article/{zim}/{path} (Reader.svelte) and GET /api/zims/{id}/random
// (the archive-browse page's "Random article" action) — same response shape.
export interface ArticleSection {
	heading_path: string[];
	level: number;
	char_start: number;
	char_end: number;
}

export interface ArticleOut {
	zim_id: number;
	path: string;
	title: string;
	text: string;
	sections: ArticleSection[];
	flags: number;
	/** 0008: media assets for media-ZIM entries (native <video> in the Reader). */
	media?: MediaOut | null;
	/** 0013: document manifest for documents-kind entry hits (manifest title lives here). */
	document?: DocumentOut | null;
}

// ── Advanced → Eval (mirrors src/vesta/api/eval.py) ──────────────────────────

export interface EvalRunResponse {
	id: number;
	profile: string;
	profile_hash: string;
	status: string; // running | done | error
}

/** RunMetrics.to_dict() per-slice shape (src/vesta/eval/metrics.py SliceMetrics). */
export interface EvalSliceMetrics {
	count: number;
	'recall@1': number;
	'recall@5': number;
	'recall@10': number;
	'recall@20': number;
	'ndcg@10': number;
	mrr: number;
}

export interface EvalLatencyPercentiles {
	stage_p50_ms: Record<string, number>;
	stage_p95_ms: Record<string, number>;
	total_p50_ms: number;
	total_p95_ms: number;
}

/** RunMetrics.to_dict(). Retrieval-only — no citation-precision/refusal-rate
 * fields exist here (those belong to the answer benchmark, not eval). */
export interface EvalRunMetrics {
	query_count: number;
	degraded: boolean;
	degraded_components: string[];
	slices: Record<string, EvalSliceMetrics>;
	latency_ms: EvalLatencyPercentiles;
}

export interface EvalQueryResult {
	id: string;
	query: string;
	slice: string;
	expected_paths: string[];
	retrieved_paths: string[];
	hit_rank: number | null;
	expected_fact: string | null;
	provenance: string | null;
}

/** RunRecord.to_metrics_json() — EvalRunDetail.metrics is this wrapper, not
 * EvalRunMetrics directly (see api/eval.py `_to_detail`). */
export interface EvalMetricsBlob {
	metrics: EvalRunMetrics;
	per_query: EvalQueryResult[];
}

export interface EvalRunConfig {
	profile_name: string;
	profile_hash: string;
	profile_yaml: string;
	golden_hash: string;
	archive_path: string;
	archive_checksum: string;
	settings_snapshot: Record<string, unknown>;
	git_sha: string;
	machine_id: string;
	notes: string;
}

/** GET /api/eval/runs returns list[EvalRunDetail] directly — no separate
 * summary DTO for eval (unlike benchmark). */
export interface EvalRunDetail {
	id: number;
	started_at: string;
	profile: string;
	profile_hash: string;
	golden_hash: string;
	archive_checksum: string;
	git_sha: string;
	machine_id: string;
	status: string;
	metrics: EvalMetricsBlob;
	config: EvalRunConfig;
}

// ── Advanced → Benchmarks (mirrors src/vesta/api/bench.py) ──────────────────

export interface BenchRunSummary {
	id: number;
	run_group: string;
	label: string;
	started_at: string;
	finished_at: string | null;
	status: string;
	dataset_name: string;
	dataset_hash: string;
	subset_hash: string;
	system: string;
	profile_name: string;
	answer_model: string;
	judge_model: string;
	scope: string;
	trusted: boolean;
	headroom: number | null;
	strict_accuracy: number | null;
	source_recall_at_10: number | null;
	hallucination_rate: number | null;
	unjudged: number | null;
	complete: number | null;
}

/** POST /api/bench/run response — the created run group + run ids (job-shaped). */
export interface BenchRunResponse {
	run_group: string;
	run_ids: number[];
	systems: string[];
	profiles: string[];
	models: string[];
	repeats: number;
	matrix_size: number;
	dataset_name: string;
	dataset_hash: string;
	subset_hash: string;
	judge_model: string;
	status: string;
}

/** The 2x2 failure-attribution cell names (mirror api/bench.py attribution keys). */
export type AttributionCell =
	| 'correct_source_found'
	| 'correct_source_missed'
	| 'failed_source_found'
	| 'failed_source_missed';

export interface BenchAttributionCounts {
	correct_source_found: number;
	correct_source_missed: number;
	failed_source_found: number;
	failed_source_missed: number;
}

export interface BenchByCapability {
	n: number;
	source_recall_at_10: number;
	source_coverage: number;
	strict_accuracy: number;
	weighted_accuracy: number;
	attribution: BenchAttributionCounts;
}

/** The metrics_json blob carried by BenchRunDetail.metrics (bench_runner._compute_metrics). */
export interface BenchMetrics {
	source: {
		n: number;
		recall_at_1: number;
		recall_at_5: number;
		recall_at_10: number;
		recall_at_20: number;
		source_coverage: number;
		source_mrr: number;
		retrieved_precision: number;
	};
	answer: {
		n: number;
		strict_accuracy: number;
		weighted_accuracy: number;
		sub_fact_coverage: number;
		abstention_correctness: number;
		over_refusal: number;
		hallucination_rate: number;
		unjudged: number;
		complete: number;
	};
	reference: {
		ceiling: number;
		system: number;
		floor: number;
		total: number;
		headroom_realised: number | null;
		retrieval_regressions: number;
		suppressed_reason: string;
	};
	attribution: BenchAttributionCounts;
	by_capability: Record<string, BenchByCapability>;
}

export interface BenchRunDetail {
	id: number;
	run_group: string;
	label: string;
	started_at: string;
	finished_at: string | null;
	status: string;
	dataset_name: string;
	dataset_hash: string;
	subset_hash: string;
	system: string;
	profile_name: string;
	profile_hash: string;
	answer_model: string;
	judge_model: string;
	scope: string;
	trusted: boolean;
	calibration: number | null;
	judge_shares_endpoint: boolean;
	abort_reason: string;
	config_json: Record<string, unknown>;
	metrics: BenchMetrics;
	progress: Record<string, unknown> | null;
}

export interface BenchResultRow {
	run_id: number;
	question_id: string;
	capability: string;
	difficulty: string;
	question_text: string;
	expected_answer: string;
	answer_text: string;
	abstained: boolean;
	verdict: string;
	verdict_reason: string;
	source_hit_rank: number | null;
	source_coverage: number;
	sub_fact_coverage: number | null;
	retrieved_paths: string[];
	rounds: number;
	latency_ms: number;
	error: string | null;
}

export interface BenchResultsPage {
	items: BenchResultRow[];
	total: number;
	offset: number;
	limit: number;
}

export interface BenchComparePair {
	run_a: number;
	run_b: number;
	shared_denominator: number;
	fixed: string[];
	broken: string[];
	both_correct: string[];
	both_wrong: string[];
	only_a: string[];
	only_b: string[];
	deltas: Record<string, number>;
}

export interface BenchCompareResponse {
	runs: number[];
	pairs: BenchComparePair[];
}

export interface BenchDatasetInfo {
	name: string;
	version: number;
	hash: string;
	generated: string;
	total: number;
	by_capability: Record<string, number>;
	by_difficulty: Record<string, number>;
	by_slice: Record<string, number>;
	ceiling: { correct: number; total: number; score: number };
	floor: { correct: number; total: number; score: number };
}

// ── Retrieval profiles (mirrors src/vesta/api/retrieval.py) ──────────────────

export interface ProfileItem {
	name: string;
	description: string;
	hash: string;
	builtin: boolean;
}

// ── Settings (mirrors src/vesta/api/settings.py's SettingSchemaOut) ──────────
// GET /api/settings/schema -> {settings: SettingSchemaItem[]}. GET /api/settings
// -> {values: {key: value}} where each value is the setting's *native* JSON
// type (bool/number/string — verified live, not a string despite the PUT
// direction requiring one; see settings-basic.ts's header comment and the
// handoff doc's gotcha list). PUT /api/settings requires every value as a
// string; `min`/`max` are always numbers (backend emits them as float even
// for integer settings) or null.

export type SettingValueType = 'boolean' | 'integer' | 'float' | 'string';

export interface SettingSchemaItem {
	key: string;
	type: SettingValueType;
	default: string | number | boolean | null;
	group: string;
	help: string;
	min: number | null;
	max: number | null;
	choices: string[] | null;
	hot: boolean;
}
