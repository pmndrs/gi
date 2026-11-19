/**
 * The panel's "Radiance cache · compare" section (RC4.5): every compare
 * toggle of the radiance cache, grouped (screen probes, world cache,
 * denoiser, experiments), each with its hover help saying what it does and
 * which value is Capsaicin's. Drawn from the core's registry
 * (`RADIANCE_CACHE_COMPARE_TOGGLES`), so a later branch adds a toggle there
 * and it appears here, in the links and in the tests with nothing else to
 * change. A small readout shows how much the GI changes per frame and the
 * frames since the last toggle change.
 *
 * Like the other panel parts it only builds controls and reports what the
 * user did; `ViewerApp` decides what that means.
 */

import {
    capsaicinCompareValues,
    COMPARE_GROUP_LABELS,
    COMPARE_GROUPS,
    compareTogglesIn,
    type CompareToggle,
    type CompareValues,
} from '../../../../packages/ddgi-core/src/techniques/radiance-cache/compare-toggles.ts';
import { h } from './dom';
import { withHelp } from './panel';

export interface RcCompareHandlers {
    /** The values that differ from the defaults, after any control changed. */
    onCompare(values: CompareValues): void;
}

/** What the readout shows; null fields are not known yet. */
export interface RcCompareReadout {
    /** Mean relative change of the GI term per pixel from the previous frame, in percent. */
    meanChangePercent: number | null;
    /** Frames drawn since a toggle last changed. */
    framesSinceChange: number;
}

type Value = boolean | number | string;

/** One control per toggle, and how to show a value in it. */
interface ToggleControl {
    readonly entry: CompareToggle;
    readonly row: HTMLElement;
    show(value: Value): void;
}

export class RcCompareControls {
    readonly section = h('section');

    private readonly controls: ToggleControl[] = [];
    private readonly readout = h('div', { className: 'value', text: '—' });
    private values: CompareValues = {};

    constructor(private readonly handlers: RcCompareHandlers) {
        const capsaicin = withHelp(h('button', { text: 'All Capsaicin' }), 'rcCompareCapsaicin');
        capsaicin.dataset.compare = 'all-capsaicin';
        capsaicin.addEventListener('click', () => this.set(capsaicinCompareValues()));
        const ours = withHelp(h('button', { text: 'All ours' }), 'rcCompareOurs');
        ours.dataset.compare = 'all-ours';
        ours.addEventListener('click', () => this.set({}));

        this.section.append(
            withHelp(h('h2', { text: 'Radiance cache · compare' }), 'rcCompare'),
            h('div', { className: 'button-row' }, capsaicin, ours),
            withHelp(h('div', { className: 'label' }, 'GI change per frame ', this.readout), 'rcCompareReadout'),
        );
        for (const group of COMPARE_GROUPS) {
            const entries = compareTogglesIn(group);
            if (entries.length === 0) continue;
            this.section.append(h('div', { className: 'label', text: COMPARE_GROUP_LABELS[group] }));
            for (const entry of entries) {
                const control = this.buildControl(entry);
                this.controls.push(control);
                this.section.append(control.row);
            }
        }
        this.showValues();
    }

    /** Shows `values` without firing the handler (the link's values, at start). */
    setValues(values: CompareValues): void {
        this.values = { ...values };
        this.showValues();
    }

    /** Hidden unless the radiance cache is on screen. */
    setVisible(visible: boolean): void {
        this.section.hidden = !visible;
    }

    setReadout(readout: RcCompareReadout | null): void {
        if (!readout) {
            this.readout.textContent = '—';
            return;
        }
        const change = readout.meanChangePercent === null ? '…' : `${readout.meanChangePercent.toFixed(3)}%`;
        this.readout.textContent = `${change} · ${readout.framesSinceChange} frames since a toggle changed`;
    }

    private set(values: CompareValues): void {
        this.values = { ...values };
        this.showValues();
        this.handlers.onCompare(this.values);
    }

    private changed(entry: CompareToggle, value: Value): void {
        const next = { ...this.values } as Record<string, Value>;
        if (value === entry.defaultValue) delete next[entry.setting];
        else next[entry.setting] = value;
        this.set(next as CompareValues);
    }

    private showValues(): void {
        const values = this.values as Record<string, Value | undefined>;
        for (const control of this.controls) {
            const value = values[control.entry.setting] ?? control.entry.defaultValue;
            control.show(value);
            // Marks a toggle that differs from ours, so a mixed state reads at a glance.
            control.row.classList.toggle('changed', value !== control.entry.defaultValue);
        }
    }

    private buildControl(entry: CompareToggle): ToggleControl {
        const capsaicinText = entry.capsaicin === undefined ? null : describeValue(entry, entry.capsaicin);
        const title = `${entry.label}${entry.resetsHistory || entry.restartsTechnique ? ' ↺' : ''}`;
        const help = `${entry.help}${entry.register ? ` (Register ${entry.register}.)` : ''}${entry.resetsHistory ? ' Changing it starts the probes over (↺).' : ''}${entry.restartsTechnique ? ' Changing it starts the radiance cache over (↺).' : ''}`;
        const withEntryHelp = <T extends HTMLElement>(element: T): T => {
            element.dataset.help = help;
            element.title = help;
            element.dataset.compareSetting = entry.setting;
            return element;
        };
        switch (entry.kind) {
            case 'boolean': {
                const input = h('input', { type: 'checkbox' });
                input.addEventListener('change', () => this.changed(entry, input.checked));
                // Its own line: inline labels would otherwise flow side by side.
                const row = withEntryHelp(h('div', {}, h('label', { className: 'inline' }, input, ` ${title}`, h('span', { className: 'muted small', text: capsaicinText === null ? ' (not in Capsaicin)' : ` (Capsaicin: ${capsaicinText})` }))));
                return { entry, row, show: (value) => (input.checked = value === true) };
            }
            case 'choice': {
                const select = h('select');
                for (const choice of entry.choices) select.append(h('option', { value: choice.value, text: choice.label }));
                select.addEventListener('change', () => this.changed(entry, select.value));
                const row = withEntryHelp(h('div', {}, h('div', { className: 'label small', text: title }), select));
                return { entry, row, show: (value) => (select.value = String(value)) };
            }
            case 'number': {
                const input = h('input', { type: 'range', min: entry.min, max: entry.max, step: entry.step });
                const shown = h('span', { className: 'value' });
                const showNumber = (value: number) => (shown.textContent = String(value));
                input.addEventListener('input', () => showNumber(Number(input.value)));
                // On release, so dragging doesn't reset or reconfigure every pixel of the way.
                input.addEventListener('change', () => this.changed(entry, Number(input.value)));
                const row = withEntryHelp(
                    h(
                        'div',
                        {},
                        h('div', { className: 'label small' }, `${title} `, shown, h('span', { className: 'muted small', text: ` (ours ${describeValue(entry, entry.defaultValue)}${capsaicinText === null ? '' : `, Capsaicin ${capsaicinText}`})` })),
                        input,
                    ),
                );
                return {
                    entry,
                    row,
                    show: (value) => {
                        input.value = String(value);
                        showNumber(Number(value));
                    },
                };
            }
        }
    }
}

/** A value in the panel's words. */
function describeValue(entry: CompareToggle, value: Value): string {
    if (entry.kind === 'boolean') return value ? 'on' : 'off';
    if (entry.kind === 'choice') return entry.choices.find((choice) => choice.value === value)?.label ?? String(value);
    return String(value);
}
