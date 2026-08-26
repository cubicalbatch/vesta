// Groups `group` strings hierarchically by convention ("Retrieval / Stage B")
// Settings groups — split on `/`
// and render as section → subsection. Order groups by a small explicit list,
// then alphabetically for anything unrecognised."). Pure function, no
// component state, so the grouping logic is testable independent of Svelte.
import type { SettingSchemaItem } from './types';

export interface SettingSubsection {
	/** null = items whose `group` had no "/" — rendered flat under the section,
	 * not nested under a subsection heading. */
	name: string | null;
	items: SettingSchemaItem[];
}

export interface SettingSection {
	/** Raw top-level group string (before "/"), used as the section's key. */
	name: string;
	/** Human label — usually `name` verbatim, overridden for a couple of groups
	 * whose schema name undersells or misdescribes the section's purpose. */
	displayName: string;
	subsections: SettingSubsection[];
}

// Only a handful of top-level groups get an explicit priority slot — the
// ones a person configuring Vesta is most likely to reach for first.
// Anything not listed here (including any *new* top-level group a future
// backend change introduces) sorts alphabetically after these, which is
// exactly the "new setting needs zero frontend work" property applied to
// grouping, not just individual fields.
const SECTION_ORDER = [
	'Inference',
	'Answer',
	'Retrieval',
	'Index',
	'Passages',
	'Query',
	'Catalog',
	'Downloads',
	'Encoders',
	'Vectors',
	'Server',
	'Jobs',
	'Chat',
	'Eval',
	'Judge Inference',
	'Logging',
	'Dev Console',
	'Storage',
	'ZIM',
	'API'
];

// The schema's raw group name is sometimes an implementation label, not a
// product one. "Server" holds host/port — what a stranger would
// call "Access". This is a display-only rename; the raw group string is
// still what backs ordering and the BASIC-key lookup.
const GROUP_DISPLAY_OVERRIDES: Record<string, string> = {
	Server: 'Access'
};

export function groupSettings(schema: SettingSchemaItem[]): SettingSection[] {
	const sections = new Map<string, Map<string | null, SettingSchemaItem[]>>();

	for (const item of schema) {
		const parts = item.group.split('/').map((p) => p.trim());
		const top = parts[0];
		const sub = parts.length > 1 ? parts.slice(1).join(' / ') : null;
		if (!sections.has(top)) sections.set(top, new Map());
		const subMap = sections.get(top)!;
		if (!subMap.has(sub)) subMap.set(sub, []);
		subMap.get(sub)!.push(item);
	}

	const topNames = [...sections.keys()].sort((a, b) => {
		const ia = SECTION_ORDER.indexOf(a);
		const ib = SECTION_ORDER.indexOf(b);
		if (ia !== -1 && ib !== -1) return ia - ib;
		if (ia !== -1) return -1;
		if (ib !== -1) return 1;
		return a.localeCompare(b);
	});

	return topNames.map((top) => {
		const subMap = sections.get(top)!;
		const subNames = [...subMap.keys()].sort((a, b) => {
			if (a === null) return -1;
			if (b === null) return 1;
			return a.localeCompare(b);
		});
		return {
			name: top,
			displayName: GROUP_DISPLAY_OVERRIDES[top] ?? top,
			subsections: subNames.map((name) => ({ name, items: subMap.get(name)! }))
		};
	});
}

/** Last dot-segment of a key, humanized — the fallback label for a field
 * with no copy override (the schema has no `label`,
 * only `help`; the full key is still shown alongside, mono, so nothing is
 * hidden — this is a generic key->label transform, not per-setting code). */
export function humanizeKey(key: string): string {
	const last = key.split('.').pop() ?? key;
	const spaced = last.replace(/_/g, ' ');
	return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/** Mono font for path/url/dir/model-shaped string settings
 * control-mapping table). */
export function isMonoStringKey(key: string): boolean {
	return /(?:path|url|dir|model)$/.test(key);
}
