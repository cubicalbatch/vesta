-- Migration 0007 — ZIM content-kind classification (video / SPA / article ZIMs).
--
-- ``kind`` classifies how a ZIM's content is organized so later stages can
-- route it correctly (e.g. a ``media`` ZIM's ``text/html`` entries are
-- meta-refresh redirect stubs to a JS video player, not real articles; its
-- playable text lives in ``text/vtt`` sidecars and its asset paths in
-- ``application/json`` metadata — see plan brave-cactus S1/S4).
--
-- Classification is signal-driven (Counter mime histogram + the Kiwix ``Tags``
-- ``_videos:yes``/``_spa:yes`` convention), NEVER the scraper name, so it
-- generalises across youtube2zim / ted2zim / phet / future scrapers. The raw
-- ``Scraper`` and ``Tags`` strings are persisted alongside for transparency
-- and future use, not for branching.
--
-- Default ``'articles'`` so every existing row classifies as the kind Vesta
-- was built for; behaviour for article ZIMs is unchanged.

ALTER TABLE zims ADD COLUMN kind TEXT NOT NULL DEFAULT 'articles';
ALTER TABLE zims ADD COLUMN scraper TEXT;
ALTER TABLE zims ADD COLUMN tags TEXT;
