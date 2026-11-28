/**
 * The small bar every example page gets at the top: back to the examples,
 * the page's title, and its source on GitHub. examples/vite.config.ts
 * injects this module into every page except the landing page, so examples
 * never import it.
 */

/** Where the examples' source is browsed. The repository is pmndrs/gi. */
const SOURCE_TREE = 'https://github.com/pmndrs/gi/tree/main/examples/';

const STYLE = `
.example-nav {
    position: fixed;
    top: 10px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 100;
    display: flex;
    gap: 2px;
    padding: 3px;
    background: rgba(8, 10, 14, 0.82);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 999px;
    font: 12px/1 system-ui, sans-serif;
    color: #cfd8e3;
    white-space: nowrap;
}
.example-nav a,
.example-nav span {
    padding: 6px 10px;
    border-radius: 999px;
    color: inherit;
    text-decoration: none;
}
.example-nav a:hover {
    background: rgba(255, 200, 120, 0.14);
    color: #ffc878;
}
.example-nav .back {
    color: #ffc878;
    font-weight: 600;
}
.example-nav .title {
    color: #8b97a6;
    border-left: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 0;
}
@media (max-width: 720px) {
    .example-nav .title,
    .example-nav .source {
        display: none;
    }
}
`;

function link(className: string, href: string, text: string, external = false): HTMLAnchorElement {
    const anchor = document.createElement('a');
    anchor.className = className;
    anchor.href = href;
    anchor.textContent = text;
    if (external) {
        anchor.target = '_blank';
        anchor.rel = 'noopener';
    }
    return anchor;
}

function mount(): void {
    const base = import.meta.env.BASE_URL;
    // The page's folder under the site's root, e.g. 'ddgi/03-lights/' or 'webgpu/hello/'.
    const folder = location.pathname.startsWith(base) ? location.pathname.slice(base.length).replace(/index\.html$/, '') : '';

    const style = document.createElement('style');
    style.textContent = STYLE;
    document.head.append(style);

    const nav = document.createElement('nav');
    nav.className = 'example-nav';
    nav.setAttribute('aria-label', 'Examples');
    nav.append(link('back', base, '← Examples'));
    const title = document.createElement('span');
    title.className = 'title';
    title.textContent = document.title;
    nav.append(title);
    if (folder) nav.append(link('source', SOURCE_TREE + folder, 'Source ↗', true));
    document.body.append(nav);
}

mount();
