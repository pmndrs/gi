/** Tiny DOM helpers, so the panel reads as a description of its controls. */

type Child = Node | string | null | undefined;

export function h<K extends keyof HTMLElementTagNameMap>(
    tag: K,
    props: Partial<Record<string, string | number | boolean>> & { className?: string; text?: string } = {},
    ...children: Child[]
): HTMLElementTagNameMap[K] {
    const element = document.createElement(tag);
    for (const [name, value] of Object.entries(props)) {
        if (value === undefined) continue;
        if (name === 'className') element.className = String(value);
        else if (name === 'text') element.textContent = String(value);
        else if (typeof value === 'boolean') {
            if (value) element.setAttribute(name, '');
        } else element.setAttribute(name, String(value));
    }
    for (const child of children) {
        if (child === null || child === undefined) continue;
        element.append(child);
    }
    return element;
}
