// ISO 639-3 → English display name for the languages most represented in the
// Kiwix catalog. Offline by design — Vesta never fetches this at runtime (a
// network lookup would betray the offline-first appliance). Codes not listed
// here fall back to the raw 3-letter code; the full ISO 639-3 table is ~7.8k
// entries, so this focused map (~90 codes) covers what a user is likely to
// search for while keeping the bundle small. Mirrors the entries that dominate
// library.kiwix.org's feed.

const LANGUAGE_NAMES: Readonly<Record<string, string>> = {
	ara: 'Arabic',
	hye: 'Armenian',
	ben: 'Bengali',
	bul: 'Bulgarian',
	my: 'Burmese',
	mya: 'Burmese',
	cat: 'Catalan',
	zho: 'Chinese',
	cmn: 'Chinese (Mandarin)',
	ces: 'Czech',
	dan: 'Danish',
	nld: 'Dutch',
	eng: 'English',
	epo: 'Esperanto',
	est: 'Estonian',
	fas: 'Persian',
	fin: 'Finnish',
	fra: 'French',
	fry: 'Frisian',
	glg: 'Galician',
	kat: 'Georgian',
	deu: 'German',
	ell: 'Greek',
	guj: 'Gujarati',
	hat: 'Haitian Creole',
	hau: 'Hausa',
	hbo: 'Ancient Hebrew',
	hbs: 'Serbo-Croatian',
	heb: 'Hebrew',
	hin: 'Hindi',
	hun: 'Hungarian',
	ina: 'Interlingua',
	ind: 'Indonesian',
	ile: 'Interlingue',
	iku: 'Inuktitut',
	gle: 'Irish',
	is: 'Icelandic',
	isl: 'Icelandic',
	ita: 'Italian',
	jpn: 'Japanese',
	kan: 'Kannada',
	kas: 'Kashmiri',
	kaz: 'Kazakh',
	khm: 'Khmer',
	kur: 'Kurdish',
	lav: 'Latvian',
	lat: 'Latin',
	lit: 'Lithuanian',
	ltz: 'Luxembourgish',
	mlg: 'Malagasy',
	msa: 'Malay',
	mal: 'Malayalam',
	mlt: 'Maltese',
	mri: 'Maori',
	mar: 'Marathi',
	mkd: 'Macedonian',
	mng: 'Mongolian',
	nep: 'Nepali',
	nob: 'Norwegian (Bokmål)',
	nno: 'Norwegian (Nynorsk)',
	nor: 'Norwegian',
	oci: 'Occitan',
	ori: 'Odia',
	pus: 'Pashto',
	pol: 'Polish',
	por: 'Portuguese',
	pan: 'Punjabi',
	roh: 'Romansh',
	rus: 'Russian',
	sin: 'Sinhala',
	slk: 'Slovak',
	slv: 'Slovenian',
	sna: 'Shona',
	som: 'Somali',
	crp: 'Seychelles Creole',
	spa: 'Spanish',
	sqi: 'Albanian',
	srp: 'Serbian',
	sun: 'Sundanese',
	swa: 'Swahili',
	swe: 'Swedish',
	tgl: 'Tagalog',
	tam: 'Tamil',
	tat: 'Tatar',
	tel: 'Telugu',
	tha: 'Thai',
	bod: 'Tibetan',
	tur: 'Turkish',
	ukr: 'Ukrainian',
	urd: 'Urdu',
	uzb: 'Uzbek',
	vie: 'Vietnamese',
	cym: 'Welsh',
	yid: 'Yiddish',
	yor: 'Yoruba',
	zul: 'Zulu'
};

/** English display name for an ISO 639-3 code, falling back to the raw code
 * (so an unmapped language is still identifiable — never blank). */
export function languageName(code: string): string {
	return LANGUAGE_NAMES[code] ?? code;
}

/** Splits a comma-separated catalog language field (some multilingual archives
 * ship dozens of codes), maps each to its name, and caps the list — the first
 * few by name plus a remainder count. Mirrors `formatLanguages`'s shape but with
 * names; the full list stays available via a `title` attribute on the element. */
export function formatLanguageNames(language: string, max = 3): string {
	const codes = language
		.split(',')
		.map((c) => c.trim())
		.filter(Boolean);
	if (codes.length <= max) return codes.map(languageName).join(', ');
	return `${codes.slice(0, max).map(languageName).join(', ')} +${codes.length - max}`;
}
