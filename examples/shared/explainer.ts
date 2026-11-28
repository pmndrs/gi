/**
 * The explainer panel every example page has (docs/plans/examples.md, "What
 * every page has"): a collapsible box, top left, with four parts.
 *
 * - **What you're seeing:** two or three sentences.
 * - **How it works:** a short mechanism, with an optional inline SVG.
 * - **Try this:** two to four numbered steps, each naming a control and what to watch for.
 * - **Trade-offs:** pros and cons, every number with its source next to it.
 *
 * Then links to the method's API page and to the guide
 * (docs/guides/ddgi-or-radiance-cache.md).
 *
 * The content is plain data, so an example keeps it out of its `main.ts`:
 * each page has an `about.ts` beside it that calls `showExplainer` with its
 * text, and loads it from its `index.html` before `main.ts`.
 *
 * Text may hold `code` in backticks and [links](https://…); nothing else is
 * read as markup. The diagram is SVG markup, set as it is: it is the page's
 * own, never a visitor's.
 *
 * The panel takes the place of the page's `.info` box, if it has one: the
 * box's children move under "What you're seeing", so a live line there (08's
 * `#bake-state`) keeps working. It ends above the stats line, and starts
 * closed on a phone-width screen.
 */

import './explainer.css';

/** A link: what it says, and where it goes. */
export interface ExplainerLink {
    readonly label: string;
    readonly href: string;
}

/** One pro or con. A number in it needs its source: a docs page or a DECISIONS entry. */
export interface TradeOff {
    readonly text: string;
    readonly source?: ExplainerLink;
}

export interface ExplainerContent {
    /** What you're seeing: two or three sentences, one paragraph each or together. */
    readonly seeing: readonly string[];
    /** How it works: short paragraphs. */
    readonly how: readonly string[];
    /** One small diagram for "How it works", as inline SVG markup. Use `currentColor` so it follows the panel's colours. */
    readonly diagram?: string;
    /** A line under the diagram. */
    readonly diagramCaption?: string;
    /** Try this: two to four steps, each naming a control and what to watch for. */
    readonly tryThis: readonly string[];
    /** Trade-offs, for and against. Left out, the section isn't shown. */
    readonly tradeOffs?: { readonly pros: readonly TradeOff[]; readonly cons: readonly TradeOff[] };
    /** The method's API page (see {@link API_PAGES}). */
    readonly api: ExplainerLink;
    /** Further links, after the API page and the guide. */
    readonly more?: readonly ExplainerLink[];
}

/** Where the docs are read: the repository's files on GitHub, so the published site links work too. */
export const DOCS_ROOT = 'https://github.com/pmndrs/gi/blob/main/docs/';

/** A link into `docs/`: `docsLink('api/three.md#ddgivolume', 'DDGIVolume')`. */
export function docsLink(path: string, label: string): ExplainerLink {
    return { label, href: DOCS_ROOT + path };
}

/** The methods' API pages. */
export const API_PAGES = {
    /** DDGI through three.js: `DDGIVolume`, `ddgi()`, `ThreeGI`. */
    three: docsLink('api/three.md', 'three.js API'),
    /** The platform and DDGI without three.js. */
    platform: docsLink('api/platform.md', 'Platform API'),
    /** The radiance cache. */
    radianceCache: docsLink('api/radiance-cache.md', 'Radiance cache API'),
} as const;

/** The guide every page links: which method for what. */
export const GUIDE = docsLink('guides/ddgi-or-radiance-cache.md', 'DDGI or the radiance cache?');

/** A piece of inline text: plain, `code`, or a link. */
export type InlinePart = { readonly kind: 'text'; readonly text: string } | { readonly kind: 'code'; readonly text: string } | { readonly kind: 'link'; readonly text: string; readonly href: string };

const INLINE = /`([^`]+)`|\[([^\]]+)\]\(([^)\s]+)\)/g;

/** Splits text into plain runs, `code` in backticks and [links](href). Exported for its test. */
export function parseInline(text: string): InlinePart[] {
    const parts: InlinePart[] = [];
    let last = 0;
    for (const match of text.matchAll(INLINE)) {
        const index = match.index ?? 0;
        if (index > last) parts.push({ kind: 'text', text: text.slice(last, index) });
        if (match[1] !== undefined) parts.push({ kind: 'code', text: match[1] });
        else parts.push({ kind: 'link', text: match[2], href: match[3] });
        last = index + match[0].length;
    }
    if (last < text.length) parts.push({ kind: 'text', text: text.slice(last) });
    return parts;
}

function anchor(link: ExplainerLink): HTMLAnchorElement {
    const element = document.createElement('a');
    element.href = link.href;
    element.textContent = link.label;
    if (/^https?:/.test(link.href)) {
        element.target = '_blank';
        element.rel = 'noopener';
    }
    return element;
}

/** `text` as DOM nodes, appended to `parent`. */
function appendInline(parent: HTMLElement, text: string): void {
    for (const part of parseInline(text)) {
        if (part.kind === 'text') parent.append(part.text);
        else if (part.kind === 'code') {
            const code = document.createElement('code');
            code.textContent = part.text;
            parent.append(code);
        } else parent.append(anchor({ label: part.text, href: part.href }));
    }
}

function element<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string, text?: string): HTMLElementTagNameMap[K] {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) appendInline(node, text);
    return node;
}

function section(title: string, open: boolean): { details: HTMLDetailsElement; body: HTMLElement } {
    const details = element('details', 'explainer-section');
    details.open = open;
    details.append(element('summary', undefined, title));
    const body = element('div', 'explainer-section-body');
    details.append(body);
    return { details, body };
}

export interface Explainer {
    /** The panel. */
    readonly element: HTMLElement;
    /** The "What you're seeing" section's body: append live lines here. */
    readonly seeing: HTMLElement;
    /** Whether the panel is open (true) or folded to its button. */
    open: boolean;
}

/** Below this width the panel starts closed (the nav hides its title at the same width). */
const NARROW_SCREEN_PX = 720;
const STORAGE_KEY = 'gi-examples:explainer-open';

function rememberedOpen(): boolean | null {
    try {
        const value = localStorage.getItem(STORAGE_KEY);
        return value === null ? null : value === '1';
    } catch {
        return null;
    }
}

function rememberOpen(open: boolean): void {
    try {
        localStorage.setItem(STORAGE_KEY, open ? '1' : '0');
    } catch {
        // Storage is a convenience; without it the panel just doesn't remember.
    }
}

/** Shows the explainer panel for this page. Call it once, before the example's own code runs. */
export function showExplainer(content: ExplainerContent): Explainer {
    const panel = element('aside', 'explainer');
    panel.setAttribute('aria-label', 'About this example');
    const body = element('div', 'explainer-body');
    body.id = 'explainer-body';

    const toggle = element('button', 'explainer-toggle');
    toggle.type = 'button';
    toggle.setAttribute('aria-controls', body.id);
    const toggleLabel = element('span', undefined, 'About this example');
    const chevron = element('span', 'explainer-chevron');
    chevron.setAttribute('aria-hidden', 'true');
    toggle.append(toggleLabel, chevron);

    const seeing = section("What you're seeing", true);
    for (const paragraph of content.seeing) seeing.body.append(element('p', undefined, paragraph));
    // The page's own note box, if any, moves in here, its live lines with it.
    const info = document.querySelector<HTMLElement>('.info');
    if (info) {
        seeing.body.append(...info.childNodes);
        info.remove();
    }
    body.append(seeing.details);

    const how = section('How it works', false);
    for (const paragraph of content.how) how.body.append(element('p', undefined, paragraph));
    if (content.diagram) {
        const figure = element('figure', 'explainer-diagram');
        // The page's own SVG, never a visitor's: set as markup.
        figure.innerHTML = content.diagram;
        if (content.diagramCaption) figure.append(element('figcaption', undefined, content.diagramCaption));
        how.body.append(figure);
    }
    body.append(how.details);

    const tryThis = section('Try this', false);
    const steps = element('ol', 'explainer-steps');
    for (const step of content.tryThis) steps.append(element('li', undefined, step));
    tryThis.body.append(steps);
    body.append(tryThis.details);

    if (content.tradeOffs) {
        const tradeOffs = section('Trade-offs', false);
        const table = element('table', 'explainer-tradeoffs');
        const rows: Array<[TradeOff, 'pro' | 'con']> = [...content.tradeOffs.pros.map((item): [TradeOff, 'pro'] => [item, 'pro']), ...content.tradeOffs.cons.map((item): [TradeOff, 'con'] => [item, 'con'])];
        for (const [item, kind] of rows) {
            const row = element('tr', kind);
            const sign = element('td', 'sign', kind === 'pro' ? '+' : '−');
            sign.setAttribute('aria-label', kind === 'pro' ? 'For' : 'Against');
            const text = element('td', 'text', item.text);
            if (item.source) {
                const source = element('span', 'source');
                source.append('Source: ', anchor(item.source));
                text.append(source);
            }
            row.append(sign, text);
            table.append(row);
        }
        tradeOffs.body.append(table);
        body.append(tradeOffs.details);
    }

    const links = element('p', 'explainer-links');
    for (const [index, link] of [content.api, GUIDE, ...(content.more ?? [])].entries()) {
        if (index > 0) links.append(' · ');
        links.append(anchor(link));
    }
    body.append(links);

    panel.append(toggle, body);
    document.body.append(panel);

    const setOpen = (open: boolean): void => {
        panel.classList.toggle('closed', !open);
        toggle.setAttribute('aria-expanded', String(open));
        toggle.title = open ? 'Fold the explainer' : 'Open the explainer';
    };
    let isOpen = rememberedOpen() ?? window.innerWidth > NARROW_SCREEN_PX;
    setOpen(isOpen);
    toggle.addEventListener('click', () => {
        isOpen = !isOpen;
        setOpen(isOpen);
        rememberOpen(isOpen);
    });

    return {
        element: panel,
        seeing: seeing.body,
        get open() {
            return isOpen;
        },
        set open(value: boolean) {
            isOpen = value;
            setOpen(value);
        },
    };
}
