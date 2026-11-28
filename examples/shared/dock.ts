/**
 * The column, bottom right above the stats line, that floating panels share
 * (timeline.ts, pass-timing.ts), so a page with several stacks them instead
 * of drawing one over another. A panel given a container of its own doesn't
 * use it.
 */

import './dock.css';

let dock: HTMLElement | null = null;

/** The page's dock, made on first use. */
export function floatingDock(): HTMLElement {
    if (!dock || !dock.isConnected) {
        dock = document.createElement('div');
        dock.className = 'floating-dock';
        document.body.append(dock);
    }
    return dock;
}
