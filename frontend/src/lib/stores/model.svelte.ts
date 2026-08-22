// The LLM lifecycle store. Polls GET /api/models/status
// every 5 s while the tab is visible and stops when hidden — status() is
// read-only server-side (it never stamps last_used), so the visibility gate
// is politeness, not correctness. Anything user-initiated calls refresh()
// directly: activate, load, unload, download completion, answer finished.
import { modelsApi, type LlmStatus } from '$lib/api/models';

const POLL_MS = 5_000;

// The chip's state→presentation mapping, pure so it can
// be unit-tested without a DOM. `popover: false` means the chip is a plain
// link to Settings → AI instead of opening the popover.
export interface ChipView {
	dot: 'hollow' | 'green' | 'amber' | 'red';
	pulsing: boolean;
	label: string;
	popover: boolean;
}

export function chipView(s: LlmStatus | null): ChipView {
	if (!s) return { dot: 'hollow', pulsing: false, label: '…', popover: false };
	switch (s.state) {
		case 'loaded':
			return { dot: 'green', pulsing: false, label: s.display_name ?? 'Model', popover: true };
		case 'loading':
			return {
				dot: 'amber',
				pulsing: true,
				label: `Loading ${s.display_name ?? 'model'}…`,
				popover: true
			};
		case 'error':
			return { dot: 'red', pulsing: false, label: 'AI unavailable', popover: true };
		case 'unloaded':
		case 'sleeping':
		case 'stopped':
			return {
				dot: 'hollow',
				pulsing: false,
				label: `${s.display_name ?? 'Model'} · asleep`,
				popover: true
			};
		default:
			// absent — and anything unmapped reads as "no AI configured".
			return { dot: 'hollow', pulsing: false, label: 'No AI', popover: false };
	}
}

class ModelStore {
	status = $state<LlmStatus | null>(null);
	loaded = $state(false); // first fetch completed (ok or not)
	error = $state<string | null>(null);
	// Which lifecycle action is mid-flight — Load/Unload buttons disable on it.
	busy = $state<'load' | 'unload' | null>(null);

	// window.setInterval id — number per the DOM lib, null when not polling.
	#timer: number | null = null;
	#inFlight = false;
	#running = false;

	/** Idempotent. The ModelChip's mount effect calls this; pages may too. */
	start(): void {
		if (this.#running) return;
		this.#running = true;
		document.addEventListener('visibilitychange', this.#onVisibility);
		if (document.visibilityState === 'visible') this.#startTimer();
	}

	stop(): void {
		if (!this.#running) return;
		this.#running = false;
		document.removeEventListener('visibilitychange', this.#onVisibility);
		this.#stopTimer();
	}

	/** Re-fetch now — event-driven updates (D9), not just the poll. */
	async refresh(): Promise<void> {
		if (this.#inFlight) return;
		this.#inFlight = true;
		try {
			this.status = await modelsApi.status();
			this.error = null;
		} catch (err) {
			this.error = err instanceof Error ? err.message : 'failed to load model status';
		} finally {
			this.#inFlight = false;
			this.loaded = true;
		}
	}

	async loadModel(): Promise<void> {
		this.busy = 'load';
		try {
			this.status = await modelsApi.load(); // blocks server-side until loaded
		} catch (err) {
			this.error = err instanceof Error ? err.message : 'load failed';
		} finally {
			this.busy = null;
			await this.refresh();
		}
	}

	async unloadModel(): Promise<void> {
		this.busy = 'unload';
		try {
			this.status = await modelsApi.unload();
		} catch (err) {
			this.error = err instanceof Error ? err.message : 'unload failed';
		} finally {
			this.busy = null;
			await this.refresh();
		}
	}

	#onVisibility = (): void => {
		if (document.visibilityState === 'visible') {
			this.#startTimer();
			void this.refresh(); // catch up immediately on return
		} else {
			this.#stopTimer();
		}
	};

	#startTimer(): void {
		if (this.#timer !== null) return;
		void this.refresh();
		this.#timer = window.setInterval(() => void this.refresh(), POLL_MS);
	}

	#stopTimer(): void {
		if (this.#timer === null) return;
		window.clearInterval(this.#timer);
		this.#timer = null;
	}
}

export const modelStore = new ModelStore();
