<script lang="ts">
	// SearchPage — the orchestrator for the unified search surface
	// Search page component. Owns mode,
	// query, scope, the sources result, the AnswerSession, and EVERY URL write.
	//
	// The two pipelines already exist behind the API; this component decides
	// which one a submitted query hits, based solely on the `Use AI` toggle:
	//   - off (sources) → GET /api/search   (ranked cards + trace)
	//   - on  (ai)       → POST /api/chat    (streaming cited answer)
	//
	// The hard part is the self-write guard (Trap 1): the page must react to
	// URL changes it did NOT make (a nav link, a goto from a source card or the
	// reader — same route, no remount) while ignoring the ones it DID make
	// (the shallow pushState after a first turn, which would remount-and-kill
	// the in-flight stream if it re-seeded). See `lastSelfWrittenUrl`.
	import { onMount, untrack } from 'svelte';
	import { page } from '$app/state';
	import { goto, pushState } from '$app/navigation';
	import type { SourceCard } from '$lib/types';
	import { zimsStore } from '$lib/stores/zims.svelte';
	import { jobsStore } from '$lib/stores/jobs.svelte';
	import { healthStore } from '$lib/stores/health.svelte';
	import { readerStore } from '$lib/stores/reader.svelte';
	import { timeOfDayGreeting } from '$lib/greeting';
	import {
		parseSearchMode,
		serializeSearchMode,
		type SearchModeState
	} from '$lib/search-mode';
	import { readUseAi, writeUseAi } from '$lib/use-ai';
	import { AnswerSession } from '$lib/answer/session.svelte';
	import { PageSearch } from '$lib/page-search.svelte';
	import SearchHero from './SearchHero.svelte';
	import SourceResults from './SourceResults.svelte';
	import AnswerThread from './AnswerThread.svelte';
	import AskHistory from '$lib/components/ask/AskHistory.svelte';
	import History from '@lucide/svelte/icons/history';
	import Plus from '@lucide/svelte/icons/plus';

	type Mode = 'sources' | 'ai';

	// Mode, query and scope are component state seeded from the URL — NOT
	// $derived from it. Input value must keep showing what the user typed even
	// after the first turn drops `q` from the URL.
	//
	// Default mode comes from the stored toggle preference; an explicit `ai`
	// URL param always wins (reseedFromUrl honours that). Initialised here to
	// avoid a sources→ai flash before the first effect run.
	let mode = $state<Mode>(readUseAi() ? 'ai' : 'sources');
	let queryInput = $state('');
	let scope = $state<Set<number>>(new Set());

	// Sources-mode result region — a PageSearch: its sequence token drops a
	// slower stale response that lands after a newer query (AUDIT_0824 F3).
	const searchState = new PageSearch();

	// The AI conversation state. One session lives for the page's lifetime.
	let session = $state(new AnswerSession());
	let historyOpen = $state(false);
	let heroInputEl = $state<HTMLInputElement | null>(null);
	// Self-write guard (Trap 1). Set in the SAME statement as the goto/pushState
	// that produced a URL; the $effect below ignores any firing where the URL
	// matches this. Everything else is a genuine external navigation and re-seeds.
	let lastSelfWrittenUrl = $state<string | null>(null);

	const scopeParam = $derived(scope.size > 0 ? [...scope].join(',') : undefined);
	// AI mode with a live or restored conversation → the hero is a follow-up box.
	const inConversation = $derived(
		mode === 'ai' && (session.liveTurn != null || session.priorTurns.length > 0)
	);

	function indexJobFor(zimId: number) {
		return jobsStore.list.find(
			(j) => j.type === 'index_zim' && j.target === String(zimId) && j.status === 'running'
		);
	}
	function archiveIndexingPct(zimId: number): number | null {
		const job = indexJobFor(zimId);
		if (!job || !job.total) return null;
		return Math.round(((job.progress ?? 0) / job.total) * 100);
	}
	// Only archives whose "running" index_status is backed by an actual job (the
	// server can report running with no job after a restart; see old /+page.svelte).
	const midIndexArchive = $derived(
		zimsStore.enabled.find(
			(a) => a.index_status === 'running' && (!jobsStore.connected || indexJobFor(a.id) != null)
		)
	);

	// ── URL writes ───────────────────────────────────────────────────────────
	/** Build the SearchModeState this page currently represents. Sources mode
	 *  never carries a conversation id — it has no conversation. */
	function currentState(): SearchModeState {
		return {
			query: queryInput,
			scope: new Set(scope),
			ai: mode === 'ai',
			conversationId: mode === 'ai' ? session.conversationId : null
		};
	}

	/** Relative pathname+search key for the self-write guard — matches what we
	 *  pass to goto/pushState and what `page.url.pathname + page.url.search`
	 *  yields, without an origin mismatch (absolute vs relative). */
	function urlKey(search: string): string {
		return search ? `/${search}` : '/';
	}

	/** Write the URL with replaceState (search-mode + first-turn seed). Sets the
	 *  self-write guard so the $effect ignores the resulting change. */
	function writeUrlReplace() {
		const qs = serializeSearchMode(currentState());
		const key = urlKey(qs ? `?${qs}` : '');
		lastSelfWrittenUrl = key;
		goto(key, { replaceState: true, noScroll: true, keepFocus: true });
	}

	/** Push the conversation id shallowly (first AI turn's response). This must
	 *  NOT remount — pushState updates the address bar without a route change,
	 *  so the in-flight stream survives (Trap 1). */
	function pushConversation(id: number) {
		const state: SearchModeState = {
			query: '',
			scope: new Set(scope),
			ai: true,
			conversationId: id
		};
		const qs = serializeSearchMode(state);
		const key = urlKey(qs ? `?${qs}` : '');
		lastSelfWrittenUrl = key;
		pushState(key, {});
	}

	// Wire the session's conversation-id announcement to the shallow pushState.
	// (pushState stays in the component, not the session class.)
	session.onConversationId = pushConversation;

	// Keep the session's scope in sync as chips change (follow-up turns reuse it).
	// Hosted in an effect so it tracks scopeParam; a top-level assignment would
	// only capture the initial value.
	$effect(() => {
		session.scope = scopeParam;
	});


	// ── AI pipeline ──────────────────────────────────────────────────────────
	function startAiTurn(q: string, { asRegenerate = false }: { asRegenerate?: boolean } = {}) {
		if (!q && !asRegenerate) return;
		session.startTurn(q, { asRegenerate });
	}

	function onRegenerate() {
		if (!session.liveTurn) return;
		startAiTurn(session.liveTurn.query, { asRegenerate: true });
	}

	function onFollowUp(q: string) {
		startAiTurn(q);
		queryInput = '';
	}

	// ── Submit (the one input, two jobs) ─────────────────────────────────────
	function onSubmit(q: string) {
		if (!q) return;
		queryInput = q;
		if (mode === 'ai') {
			// First turn: write q/ai/scope to the URL (the seed), THEN start the
			// turn, THEN clear the box — it morphs into the follow-up box whose
			// "Ask a follow-up…" placeholder only shows when empty
			// "One input, one conversation control row").
			writeUrlReplace();
			startAiTurn(q);
			queryInput = '';
		} else {
			// Sources: keep the query in the box so it can be edited/re-searched.
			writeUrlReplace();
			void searchState.run(q, scopeParam);
		}
	}

	// ── Mode switching ───────────────────────────────────────────────────────
	function setMode(next: Mode) {
		if (next === mode) return;
		// Toggling ALWAYS aborts the live stream first — a stream left running
		// behind a switched-off toggle is a potential bug to
		// ship (Trap 2).
		session.abort();

		const hadQuery = queryInput.trim().length > 0;
		mode = next;
		writeUseAi(next === 'ai'); // store ONLY on a real user toggle

		if (next === 'ai' && hadQuery) {
			// sources → ai with a query starts a FRESH conversation — this is what
			// the old "Write me an answer" pill did (it remounted /ask with q),
			// so an unrelated question must never be folded into a prior thread.
			// Write q/ai/scope to the URL FIRST (the first-turn seed), then start
			// the turn, then clear the box → it morphs into the follow-up box.
			session.reset();
			writeUrlReplace();
			startAiTurn(queryInput.trim());
			queryInput = '';
		} else if (next === 'sources' && hadQuery) {
			// ai → sources with a query: same query, sources view. Keep the AI
			// session alive so toggling back restores the thread. Sources mode
			// drops `c` from the URL (it has no conversation).
			writeUrlReplace();
			void searchState.run(queryInput.trim(), scopeParam);
		} else {
			// empty query: switch mode + storage only; keep any existing thread so a
			// flip back restores it. Sources hides the thread by rendering a
			// different result region.
			writeUrlReplace();
		}
	}

	function onToggleMode() {
		setMode(mode === 'ai' ? 'sources' : 'ai');
	}
	function onToggleScope(id: number | 'all') {
		if (id === 'all') scope = new Set();
		else {
			const next = new Set(scope);
			next.has(id) ? next.delete(id) : next.add(id);
			scope = next;
		}
		// Reflect the new scope in the URL (self-write, effect ignores) and, in
		// sources mode, re-run the search for it.
		writeUrlReplace();
		if (mode === 'sources' && searchState.lastQuery)
			void searchState.run(searchState.lastQuery, scopeParam);
	}

	// ── Reader ───────────────────────────────────────────────────────────────
	function openCard(card: SourceCard, i: number) {
		readerStore.open({
			zimId: card.zim_id,
			path: card.path,
			title: card.title,
			cards: searchState.result?.cards,
			cardIndex: i
		});
	}

	// ── New question ─────────────────────────────────────────────────────────
	function newQuestion() {
		session.abort();
		session.reset();
		queryInput = '';
		searchState.clear();
		// Stay in the current mode (AI), but drop the conversation from the URL.
		writeUrlReplace();
		heroInputEl?.focus();
	}

	/** Deleting the currently-open conversation from history must drop the
	 *  session's id — a follow-up would POST it and 404 (the row is gone).
	 *  Mirrors "New question": abort, blank state, c= out of the URL. Only in
	 *  AI mode: sources mode never shows this conversation, and resetting
	 *  there would wipe results the user is looking at. */
	function conversationDeleted(id: number) {
		if (mode === 'ai' && id === session.conversationId) newQuestion();
	}

	// ── Re-seed from an external navigation ──────────────────────────────────
	/** Re-seed mode/query/scope/conversation from the URL and run whatever it
	 *  implies. Called for genuine external navigations only (the self-write
	 *  guard short-circuits our own writes). */
	function reseedFromUrl() {
		const s = parseSearchMode(page.url.search);
		// An explicit `ai` URL param always wins over the stored preference — a
		// link someone sent must land in the mode it was written for. Absent →
		// fall back to the stored toggle.
		const urlSpecifiesAi = page.url.searchParams.has('ai');
		mode = urlSpecifiesAi ? (s.ai ? 'ai' : 'sources') : (readUseAi() ? 'ai' : 'sources');
		scope = s.scope;
		session.conversationId = s.conversationId;
		session.scope = s.scope.size > 0 ? [...s.scope].join(',') : undefined;
		// Blank the surface and invalidate any in-flight search — a response
		// racing home after the navigation must not repaint the re-seeded view.
		searchState.clear();

		if (s.conversationId != null && s.ai) {
			session.reset();
			session.conversationId = s.conversationId;
			// Restored conversation → follow-up box, empty (placeholder shows).
			queryInput = '';
			void session.loadHistory(s.conversationId);
		} else if (s.ai && s.query) {
			// First turn seeded straight from a link (reader "ask about", a
			// shared `?q=&ai=1` link) — start it; the box becomes a follow-up box.
			session.reset();
			queryInput = '';
			startAiTurn(s.query);
		} else if (!s.ai && s.query) {
			// Sources: keep the query in the box (matches the old / page).
			session.reset();
			queryInput = s.query;
			void searchState.run(s.query, scopeParam);
		} else {
			session.reset();
			queryInput = s.query;
		}
	}

	// The self-write guard (Trap 1). Reads page.url so it fires on every
	// navigation; the early return skips URLs we produced ourselves (goto/
	// pushState set lastSelfWrittenUrl to this same relative key in the same
	// statement). Everything else is a genuine external navigation (nav link,
	// a goto from a source card / the reader — same route, no remount) and
	// re-seeds mode/query/scope/conversation.
	//
	// lastSelfWrittenUrl must be read `untrack`ed: it's `$state`, and reading it
	// normally would make writing it (in writeUrlReplace/pushConversation) ITSELF
	// a dependency of this effect. Since goto()/pushState are async, that write
	// re-fires this effect immediately, before page.url has actually caught up —
	// the guard then compares the new lastSelfWrittenUrl against the still-stale
	// page.url, the keys don't match, and reseedFromUrl() runs against the OLD
	// URL, stomping mode right back to what it was (this was the "Use AI" toggle
	// bug: click reads as a no-op, or a second click "undoes" the first). The
	// effect must depend on page.url alone.
	//
	// DO NOT key {#each} or the route on the search string: the conversation
	// pushState would remount mid-stream and kill the answer.
	$effect(() => {
		const key = page.url.pathname + page.url.search;
		if (untrack(() => lastSelfWrittenUrl) === key) return;
		reseedFromUrl();
	});

	// Zero-archive first run → /welcome. Trap 6: must fire on /?ai=1 too, not
	// just bare /, now that the hero is the only greeting block. Once the user
	// has finished the wizard (setup_completed), never force them back even with
	// zero archives — downloads may still be mid-flight, or they skipped
	// archives entirely. Gated on healthStore.loaded so a slow /health response
	// can't trigger a premature redirect on a reload.
	$effect(() => {
		if (
			healthStore.loaded &&
			!healthStore.data?.setup_completed &&
			zimsStore.loaded &&
			zimsStore.enabled.length === 0
		) {
			goto('/welcome');
		}
	});

	onMount(() => {
		// The one focus listener (one page, one listener — AskPage's
		// and AskTurn's are gone). `/` focuses the single query box.
		const onFocus = () => heroInputEl?.focus();
		window.addEventListener('vesta:focus-search', onFocus);
		return () => {
			window.removeEventListener('vesta:focus-search', onFocus);
			// Trap 2: abort any in-flight stream on teardown.
			session.abort();
			session.dispose();
		};
	});
</script>

<svelte:head>
	<title>Search - Vesta</title>
</svelte:head>

<div class="mx-auto mb-4 flex max-w-[var(--wide-max)] items-center justify-end gap-2">
	{#if inConversation}
		<button
			type="button"
			class="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-medium text-muted hover:bg-surface-muted hover:text-ink"
			onclick={newQuestion}
		>
			<Plus class="size-3.5" /> New question
		</button>
	{/if}
	<button
		type="button"
		class="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-medium text-muted hover:bg-surface-muted hover:text-ink"
		onclick={() => (historyOpen = true)}
	>
		<History class="size-3.5" /> History
	</button>
</div>

<AskHistory bind:open={historyOpen} onDeleted={conversationDeleted} />

<!-- Hidden once a conversation is live: AskTurn's own per-turn follow-up
     form takes over continuing it, and "New question" above brings this
     hero back. -->
{#if !inConversation}
	<SearchHero
		bind:query={queryInput}
		bind:inputEl={heroInputEl}
		mode={mode}
		greeting={timeOfDayGreeting()}
		archiveCount={zimsStore.enabled.length}
		articleCount={zimsStore.totalArticles}
		{scope}
		archives={zimsStore.enabled}
		llmAvailable={healthStore.has('llm')}
		healthLoaded={healthStore.loaded}
		indexingPct={archiveIndexingPct}
		{onSubmit}
		{onToggleMode}
		{onToggleScope}
	/>
{/if}

{#if !healthStore.has('vectors') && healthStore.loaded && mode === 'sources'}
	<p class="mx-auto mt-3 max-w-xl text-center text-xs text-faint">Meaning search is off - keyword only until an archive is indexed to depth 2+.</p>
{/if}

{#if mode === 'sources'}
	<div class="mx-auto max-w-[var(--content-max)]">
		<SourceResults
			result={searchState.result}
			loading={searchState.loading}
			error={searchState.error}
			lastQuery={searchState.lastQuery}
			midIndexArchive={midIndexArchive ?? null}
			midIndexPct={midIndexArchive ? archiveIndexingPct(midIndexArchive.id) : null}
			onOpenCard={openCard}
		/>
	</div>
{:else}
	<!-- No max-w-[var(--content-max)] wrapper here: AskTurn (inside AnswerThread)
	     has its own max-w-[var(--wide-max)] two-column grid for the sticky
	     sources rail — a narrower ancestor cramps that grid instead of letting
	     it size itself (matches the old routes/ask/AskPage.svelte). -->
	<AnswerThread {session} onRegenerate={onRegenerate} onFollowUp={onFollowUp} />
{/if}
