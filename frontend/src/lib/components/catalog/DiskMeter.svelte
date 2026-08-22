<script lang="ts">
	// Degrades to nothing if GET /api/system/storage isn't available
	// Disk meter
	// — "Degradation if rejected: drop the disk meter").
	import { systemApi, type StorageInfo } from '$lib/api/system';
	import { formatBytes } from '$lib/format';

	let info = $state<StorageInfo | null>(null);
	let unavailable = $state(false);

	$effect(() => {
		systemApi
			.storage()
			.then((s) => (info = s))
			.catch(() => (unavailable = true));
	});

	// total_bytes is 0 on some container filesystems (and on any statvfs the
	// backend couldn't read) — dividing by it yields NaN, which Tailwind emits
	// as `width: NaN%` and the browser drops, leaving a full-width bar that
	// reads as "disk full".
	const usable = $derived(info != null && info.total_bytes > 0);
	const usedPct = $derived(
		usable && info
			? Math.min(100, Math.max(0, Math.round(((info.total_bytes - info.free_bytes) / info.total_bytes) * 100)))
			: 0
	);
</script>

{#if info && usable && !unavailable}
	<div class="mt-3 max-w-xs">
		<div class="h-1.5 overflow-hidden rounded-full bg-surface-muted">
			<div class="h-full rounded-full {usedPct > 90 ? 'bg-danger' : 'bg-accent'}" style="width: {usedPct}%"></div>
		</div>
		<p class="mt-1 text-xs text-faint">{formatBytes(info.free_bytes)} free of {formatBytes(info.total_bytes)}</p>
	</div>
{/if}
