/**
 * The switcher's panel, top right: the variant, the scene and the debug
 * overlay; live readouts; the running variant's card (cards.ts); and a place
 * for the GPU time per pass.
 *
 * A variant is read when the technique is made (its world cache, the DDGI
 * volume under it, the screen channels it asks for), so picking another one
 * reloads the page with it in the link (`?variant=sharc&scene=courtyard`):
 * the old platform's memory is freed with its device, and each variant warms
 * up from empty, as P6.5 measured it. The camera is kept across the reload.
 * The overlay switches live.
 */

import type * as THREE from 'three/webgpu';
import type { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RADIANCE_CACHE_VARIANT_PRESETS, type RadianceCacheDebugViewId, type RadianceCacheVariantPreset } from '@pmndrs/gi/radiance-cache';
import { renderCard, variantCard } from './cards.ts';
import './c6.css';

export interface SwitcherChoice<Id extends string> {
    readonly id: Id;
    readonly label: string;
}

export interface SwitcherOptions<SceneId extends string> {
    readonly variant: RadianceCacheVariantPreset;
    readonly scene: SceneId;
    readonly scenes: readonly SwitcherChoice<SceneId>[];
    readonly overlays: ReadonlyArray<{ readonly id: string; readonly label: string; readonly view: RadianceCacheDebugViewId | null }>;
    readonly onOverlay: (view: RadianceCacheDebugViewId | null) => void;
}

export interface Switcher {
    /** Where the pass timing goes. */
    readonly timing: HTMLElement;
    /** Shows `lines` as the live readouts, again every half second. */
    readouts(lines: () => readonly string[]): void;
    /** A line on the card about this run (a variant that couldn't run as itself). */
    note(text: string): void;
}

function choice(select: HTMLSelectElement, value: string, label: string, selected: boolean): void {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    option.selected = selected;
    select.append(option);
}

function field(panel: HTMLElement, label: string, select: HTMLSelectElement): void {
    const row = document.createElement('label');
    row.className = 'switcher-field';
    const name = document.createElement('span');
    name.textContent = label;
    row.append(name, select);
    panel.append(row);
}

/** Reloads with `name` set in the link. */
function reloadWith(name: string, value: string): void {
    const params = new URLSearchParams(location.search);
    params.set(name, value);
    location.search = params.toString();
}

export function showSwitcher<SceneId extends string>(options: SwitcherOptions<SceneId>): Switcher {
    const panel = document.createElement('aside');
    panel.className = 'switcher';
    panel.setAttribute('aria-label', 'Cache switcher');

    const variant = document.createElement('select');
    for (const preset of RADIANCE_CACHE_VARIANT_PRESETS) {
        // The neural tail's network is an optional package the site doesn't bundle: listed, and run as the library falls back.
        const label = preset.needs.sibling ? `${preset.label} (needs the optional package)` : preset.label;
        choice(variant, preset.id, label, preset.id === options.variant.id);
    }
    variant.addEventListener('change', () => reloadWith('variant', variant.value));
    field(panel, 'Variant', variant);

    const scene = document.createElement('select');
    for (const entry of options.scenes) choice(scene, entry.id, entry.label, entry.id === options.scene);
    scene.addEventListener('change', () => reloadWith('scene', scene.value));
    field(panel, 'Scene', scene);

    const overlay = document.createElement('select');
    const initial = new URLSearchParams(location.search).get('overlay');
    for (const entry of options.overlays) choice(overlay, entry.id, entry.label, entry.id === initial);
    const applyOverlay = (): void => {
        const entry = options.overlays.find((candidate) => candidate.id === overlay.value) ?? options.overlays[0];
        options.onOverlay(entry.view);
        // Kept in the link, so a variant switch keeps it.
        const params = new URLSearchParams(location.search);
        if (entry.view === null) params.delete('overlay');
        else params.set('overlay', entry.id);
        history.replaceState(null, '', `?${params}`);
    };
    overlay.addEventListener('change', applyOverlay);
    field(panel, 'Overlay', overlay);

    const readouts = document.createElement('div');
    readouts.className = 'switcher-readouts';
    panel.append(readouts);

    const card = document.createElement('details');
    card.className = 'switcher-section';
    card.open = true;
    const cardTitle = document.createElement('summary');
    cardTitle.textContent = 'About this variant';
    const notes = document.createElement('p');
    notes.className = 'card-note';
    notes.hidden = true;
    card.append(cardTitle, notes);
    const content = variantCard(options.variant.id);
    if (content) card.append(renderCard(content));
    panel.append(card);

    const timing = document.createElement('details');
    timing.className = 'switcher-section';
    timing.open = true;
    const timingTitle = document.createElement('summary');
    timingTitle.textContent = 'This page, live';
    timing.append(timingTitle);
    panel.append(timing);

    document.body.append(panel);
    applyOverlay(); // the link's overlay, if any

    return {
        timing,
        readouts(lines) {
            const update = (): void => {
                readouts.replaceChildren(
                    ...lines().map((line) => {
                        const row = document.createElement('div');
                        row.textContent = line;
                        return row;
                    }),
                );
            };
            update();
            setInterval(update, 500);
        },
        note(text) {
            notes.textContent = text;
            notes.hidden = false;
        },
    };
}

/**
 * Keeps the camera across reloads (a variant or scene switch), per scene,
 * in session storage. Storage is a convenience: without it the camera starts
 * at the scene's bookmark.
 */
export function keepCamera(camera: THREE.PerspectiveCamera, controls: OrbitControls, scene: string): void {
    const key = `gi-examples:c6-camera:${scene}`;
    try {
        const saved = JSON.parse(sessionStorage.getItem(key) ?? 'null') as number[] | null;
        if (Array.isArray(saved) && saved.length === 6 && saved.every(Number.isFinite)) {
            camera.position.set(saved[0], saved[1], saved[2]);
            controls.target.set(saved[3], saved[4], saved[5]);
            controls.update();
        }
    } catch {
        // No storage: the bookmark's camera.
    }
    const save = (): void => {
        try {
            sessionStorage.setItem(key, JSON.stringify([...camera.position.toArray(), ...controls.target.toArray()]));
        } catch {
            // As above.
        }
    };
    controls.addEventListener('end', save);
    window.addEventListener('pagehide', save);
}
