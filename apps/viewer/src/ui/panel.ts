/**
 * The control panel. It only builds controls and reports what the user did;
 * `ViewerApp` decides what that means. Every control carries a plain-language
 * help text (see `help-text.ts`) that appears in the help box when the mouse
 * is over it.
 */

import type { CameraBookmark, SceneManifest } from '../../../../scenes/src/manifest';
import { bookmarkLabel } from '../scene/manifests';
import { TONE_MAPPERS, type ToneMappingName } from '../core/viewer-core';
import type { NavigationMode } from '../controls/camera-rig';
import { TECHNIQUE_LABELS, type GiTechniqueId, type TechniqueId, type UpscaleQuality } from '../techniques/types';
import { h } from './dom';
import { DEFAULT_HELP, HELP, type HelpKey } from './help-text';
import {
    ATLAS_EV_STEP,
    CHANGE_FULL_SCALE_PERCENTS,
    MAX_ATLAS_EV,
    MAX_BUDGET_MS,
    MAX_EXPOSURE_EV,
    MIN_BUDGET_MS,
    type AtlasName,
    type ChangeFullScalePercent,
    type DdgiScreenViewName,
    type GiChoice,
    type ProbeColorName,
    type RadianceCacheScreenViewName,
} from './url-state';

export interface DdgiPanelSettings {
    budgetMs: number;
    probes: boolean;
    probeColor: ProbeColorName;
    debug: DdgiScreenViewName | null;
    atlas: AtlasName | null;
    atlasFill: boolean;
    atlasEv: number;
    change: ChangeFullScalePercent | null;
}

export interface PanelHandlers {
    onScene(id: string): void;
    onBookmark(index: number): void;
    onCopyLink(): void;
    onNavigation(mode: NavigationMode): void;
    onFlySpeed(metresPerSecond: number): void;
    onFieldOfView(degrees: number): void;
    /** Null: back to the scene's own exposure. */
    onExposure(ev: number | null): void;
    onToneMapping(name: ToneMappingName): void;
    onReferenceCube(visible: boolean): void;
    onGi(choice: GiChoice): void;
    onTechnique(id: GiTechniqueId): void;
    onDdgiSettings(settings: DdgiPanelSettings): void;
    onDdgiRestart(): void;
    /** The radiance cache's debug view, or null for its normal picture. */
    onRadianceCacheDebug(view: RadianceCacheScreenViewName | null): void;
    onProbeSettings(settings: { bounces: number; maxProbes: number }): void;
    onRebake(): void;
    onVxgiSettings(settings: { bounces: number }): void;
    onWipe(enabled: boolean, left: TechniqueId, right: TechniqueId): void;
    onCompareWithReference(): void;
    onUpscale(quality: UpscaleQuality | null): void;
    onStats(visible: boolean): void;
    onPanelHidden(hidden: boolean): void;
}

/** One line of the probe legend: a colour, what it means, and how many probes have it (null until counted, or not counted). */
export interface ProbeLegendEntry {
    color: readonly [number, number, number];
    label: string;
    count: number | null;
}

/** Everything the panel shows, so the app can set it all in one call. */
export interface PanelState {
    sceneId: string | null;
    bookmarkIndex: number | null;
    navigation: NavigationMode;
    flySpeed: number;
    fieldOfView: number;
    /** The exposure in effect, in stops. */
    exposureEv: number;
    /** The scene's own exposure, which Reset goes back to. */
    sceneExposureEv: number;
    toneMapping: ToneMappingName;
    referenceCube: boolean;
    gi: GiChoice;
    technique: GiTechniqueId;
    ddgi: DdgiPanelSettings;
    /** The radiance cache's debug view, or null. */
    radianceCacheDebug: RadianceCacheScreenViewName | null;
    probeBounces: number;
    maxProbes: number;
    vxgiBounces: number;
    wipe: { left: TechniqueId; right: TechniqueId } | null;
    upscale: UpscaleQuality | null;
    /** False while VXGI is on screen: the upscaler can't wrap it. */
    upscaleAvailable: boolean;
    stats: boolean;
}

const UPSCALE_LABELS: Record<UpscaleQuality, string> = {
    'native-aa': 'Native (anti-aliasing only)',
    quality: 'Quality',
    balanced: 'Balanced',
    performance: 'Performance',
    'ultra-performance': 'Ultra performance',
};

const SCREEN_VIEW_LABELS: Record<DdgiScreenViewName | 'none', string> = {
    none: 'Normal picture',
    'gi-only': 'GI only, on white surfaces',
    coverage: 'Coverage (where DDGI answers)',
    'probe-states': 'Probes near this point',
    'screen-depth': 'Screen inputs: depth',
    'screen-normal': 'Screen inputs: normal',
    'screen-velocity': 'Screen inputs: motion',
    'screen-albedo': 'Screen inputs: albedo',
    'screen-instance': 'Screen inputs: object id',
};

const RADIANCE_CACHE_VIEW_LABELS: Record<RadianceCacheScreenViewName | 'none', string> = {
    none: 'Normal picture',
    'probe-atlas': 'Probe atlas (each probe, 8 × 8 directions)',
    'probe-mask': 'Probe mask (where the probes are)',
    'probe-sh': 'Probe SH (each probe as nine numbers)',
};

const PROBE_COLOR_LABELS: Record<ProbeColorName, string> = {
    state: 'State (active, off, not updated yet)',
    history: 'How settled (updates averaged)',
};

const ATLAS_LABELS: Record<AtlasName | 'none', string> = {
    none: 'Off',
    irradiance: 'Irradiance (light gathered)',
    distance: 'Distance (how far each probe sees)',
};

function changeLabel(percent: ChangeFullScalePercent | 'none'): string {
    return percent === 'none' ? 'Off (normal picture)' : `On: full colour at ${percent}% change`;
}

const PROBE_BUDGETS = [100, 300, 600, 1500, 3000];

/** The stats box's lines, as they are labelled there, with their explanations. */
const STATS_LINES: ReadonlyArray<readonly [string, HelpKey]> = [
    ['frame', 'statsFrame'],
    ['GPU per frame', 'statsGpu'],
    ['probes updated', 'statsProbes'],
    ['per frame, measured cost', 'statsBudget'],
    ['update (GPU)', 'statsDdgiUpdate'],
    ['top-level refit (GPU)', 'statsRefit'],
    ['GPU memory', 'statsMemory'],
    ['adapter (CPU)', 'statsSync'],
    ['recompiles', 'statsRecompiles'],
    ['active, off, not updated yet', 'statsStates'],
];

/** Every technique, in the order the selectors list them. */
const WIPE_CHOICES: readonly TechniqueId[] = ['ddgi', 'rc', 'lpg', 'vxgi', 'none', 'cycles'];
const GI_CHOICES: readonly GiTechniqueId[] = ['ddgi', 'rc', 'lpg', 'vxgi'];

/** The budget slider's step, in milliseconds. */
const BUDGET_STEP_MS = 0.25;


export function withHelp<T extends HTMLElement>(element: T, key: HelpKey): T {
    element.dataset.help = HELP[key];
    element.title = HELP[key];
    return element;
}

function formatEv(ev: number): string {
    return `${ev > 0 ? '+' : ''}${ev.toFixed(1)} EV`;
}

function cssColor(color: readonly [number, number, number]): string {
    // Linear to display, roughly (gamma 2.2), so the swatch looks like the spheres do on screen.
    const channel = (value: number) => Math.round(255 * Math.min(1, Math.max(0, value)) ** (1 / 2.2));
    return `rgb(${channel(color[0])}, ${channel(color[1])}, ${channel(color[2])})`;
}

export class Panel {
    readonly element: HTMLElement;

    private readonly sceneSelect = h('select');
    private readonly sceneDescription = h('p', { className: 'muted' });
    private readonly sceneCredit = h('p', { className: 'muted small' });
    private readonly bookmarkRow = h('div', { className: 'button-row' });
    private bookmarkButtons: HTMLButtonElement[] = [];

    private readonly navButtons = new Map<NavigationMode, HTMLInputElement>();
    private readonly flySpeed = h('input', { type: 'range', min: -1, max: 2.5, step: 0.01 });
    private readonly fov = h('input', { type: 'range', min: 30, max: 110, step: 1 });
    private readonly exposure = h('input', { type: 'range', min: -MAX_EXPOSURE_EV, max: MAX_EXPOSURE_EV, step: 0.1 });
    private readonly exposureValue = h('span', { className: 'value' });
    private readonly toneSelect = h('select');
    private readonly referenceCube = h('input', { type: 'checkbox' });

    private readonly giButtons = new Map<GiChoice, HTMLInputElement>();
    private readonly techniqueSelect = h('select');
    private readonly ddgiSettings = h('div', { className: 'group' });
    private readonly budget = h('input', { type: 'range', min: MIN_BUDGET_MS, max: MAX_BUDGET_MS, step: BUDGET_STEP_MS });
    private readonly budgetValue = h('span', { className: 'value' });
    private readonly showProbes = h('input', { type: 'checkbox' });
    private readonly probeColor = h('select');
    private readonly probeColorRow = h('div', { className: 'indented' });
    private readonly probeLegend = h('div', { className: 'legend' });
    private readonly screenView = h('select');
    private readonly atlas = h('select');
    private readonly atlasOptions = h('div', { className: 'indented' });
    private readonly atlasFill = h('input', { type: 'checkbox' });
    private readonly atlasEv = h('input', { type: 'range', min: -MAX_ATLAS_EV, max: MAX_ATLAS_EV, step: ATLAS_EV_STEP });
    private readonly atlasEvValue = h('span', { className: 'value' });
    private readonly atlasEvRow = h('div');
    private readonly change = h('select');
    private readonly probeSettings = h('div', { className: 'group' });
    private readonly probeBounces = h('select');
    private readonly probeBudget = h('select');
    private readonly vxgiSettings = h('div', { className: 'group' });
    private readonly radianceCacheSettings = h('div', { className: 'group' });
    private readonly radianceCacheView = h('select');
    private readonly vxgiBounces = h('select');
    private readonly status = h('div', { className: 'status' });

    private readonly wipeToggle = h('input', { type: 'checkbox' });
    private readonly wipeLeft = h('select');
    private readonly wipeRight = h('select');
    private readonly referenceNote = h('p', { className: 'muted small note' });

    private readonly upscaleToggle = h('input', { type: 'checkbox' });
    private readonly upscaleQuality = h('select');
    private readonly statsToggle = h('input', { type: 'checkbox' });
    private readonly helpBox = h('div', { className: 'help-box', text: DEFAULT_HELP });

    private sceneExposureEv = 0;

    /**
     * @param extraSections more sections (the lighting controls), placed after global illumination
     */
    constructor(
        parent: HTMLElement,
        manifests: readonly SceneManifest[],
        private readonly handlers: PanelHandlers,
        extraSections: readonly HTMLElement[] = [],
    ) {
        this.element = h('aside', { className: 'panel' });
        parent.append(this.element);

        for (const manifest of manifests) this.sceneSelect.append(h('option', { value: manifest.id, text: manifest.title }));
        for (const [name, tone] of Object.entries(TONE_MAPPERS)) this.toneSelect.append(h('option', { value: name, text: tone.label }));
        for (const bounces of [0, 1, 2]) {
            this.probeBounces.append(h('option', { value: bounces, text: `${bounces}` }));
            this.vxgiBounces.append(h('option', { value: bounces, text: `${bounces}` }));
        }
        for (const budget of PROBE_BUDGETS) this.probeBudget.append(h('option', { value: budget, text: `up to ${budget}` }));
        for (const id of GI_CHOICES) this.techniqueSelect.append(h('option', { value: id, text: TECHNIQUE_LABELS[id] }));
        for (const side of [this.wipeLeft, this.wipeRight]) {
            for (const id of WIPE_CHOICES) side.append(h('option', { value: id, text: TECHNIQUE_LABELS[id] }));
        }
        for (const [view, label] of Object.entries(SCREEN_VIEW_LABELS)) this.screenView.append(h('option', { value: view, text: label }));
        for (const [view, label] of Object.entries(RADIANCE_CACHE_VIEW_LABELS)) this.radianceCacheView.append(h('option', { value: view, text: label }));
        for (const [choice, label] of Object.entries(PROBE_COLOR_LABELS)) this.probeColor.append(h('option', { value: choice, text: label }));
        for (const [choice, label] of Object.entries(ATLAS_LABELS)) this.atlas.append(h('option', { value: choice, text: label }));
        for (const percent of ['none', ...CHANGE_FULL_SCALE_PERCENTS] as const) this.change.append(h('option', { value: String(percent), text: changeLabel(percent) }));
        for (const [quality, label] of Object.entries(UPSCALE_LABELS)) this.upscaleQuality.append(h('option', { value: quality, text: label }));

        this.element.append(
            h('h1', { text: 'DDGI viewer' }),
            this.sceneSection(),
            this.moveSection(),
            this.lightSection(),
            this.giSection(),
            ...extraSections,
            this.compareSection(),
            this.upscalerSection(),
            this.displaySection(),
            this.helpBox,
        );
        this.wireEvents();
        this.installHelp();
    }

    // -- building ----------------------------------------------------------

    private sceneSection(): HTMLElement {
        withHelp(this.sceneSelect, 'scene');
        withHelp(this.bookmarkRow, 'cameraBookmarks');
        const copy = withHelp(h('button', { text: 'Copy link to this view' }), 'copyLink');
        copy.addEventListener('click', () => this.handlers.onCopyLink());
        return h(
            'section',
            {},
            h('h2', { text: 'Scene' }),
            this.sceneSelect,
            this.sceneDescription,
            h('div', { className: 'label', text: 'Camera views' }),
            this.bookmarkRow,
            copy,
            this.sceneCredit,
        );
    }

    private moveSection(): HTMLElement {
        const radio = (mode: NavigationMode, label: string, help: HelpKey): HTMLLabelElement => {
            const input = h('input', { type: 'radio', name: 'navigation', value: mode });
            this.navButtons.set(mode, input);
            return withHelp(h('label', { className: 'inline' }, input, ` ${label}`), help);
        };
        withHelp(this.flySpeed, 'flySpeed');
        withHelp(this.fov, 'fieldOfView');
        return h(
            'section',
            {},
            h('h2', { text: 'Move' }),
            h('div', { className: 'row' }, radio('orbit', 'Orbit', 'navOrbit'), radio('fly', 'Fly', 'navFly')),
            h('div', { className: 'label', text: 'Fly speed' }),
            this.flySpeed,
            h('div', { className: 'label', text: 'Field of view' }),
            this.fov,
        );
    }

    private lightSection(): HTMLElement {
        withHelp(this.exposure, 'exposure');
        withHelp(this.toneSelect, 'toneMapping');
        const reset = withHelp(h('button', { className: 'small-button', text: 'Reset' }), 'exposureReset');
        reset.addEventListener('click', () => {
            this.exposure.value = String(this.sceneExposureEv);
            this.showExposure(this.sceneExposureEv);
            this.handlers.onExposure(null);
        });
        return h(
            'section',
            {},
            h('h2', { text: 'Light' }),
            h('div', { className: 'label' }, 'Exposure ', this.exposureValue, reset),
            this.exposure,
            h('div', { className: 'label', text: 'Tone mapping' }),
            this.toneSelect,
            withHelp(h('label', { className: 'inline' }, this.referenceCube, ' Show a 1 m reference cube'), 'referenceCube'),
        );
    }

    private giSection(): HTMLElement {
        const radio = (choice: GiChoice, label: string, help: HelpKey): HTMLLabelElement => {
            const input = h('input', { type: 'radio', name: 'gi', value: choice });
            this.giButtons.set(choice, input);
            return withHelp(h('label', { className: 'inline' }, input, ` ${label}`), help);
        };
        withHelp(this.techniqueSelect, 'technique');
        withHelp(this.budget, 'ddgiBudget');
        withHelp(this.probeLegend, 'ddgiProbes');
        withHelp(this.screenView, 'ddgiScreenView');
        withHelp(this.probeColor, 'ddgiProbeColor');
        withHelp(this.atlas, 'ddgiAtlas');
        withHelp(this.atlasEv, 'ddgiAtlasExposure');
        withHelp(this.change, 'ddgiChange');
        withHelp(this.probeBounces, 'lpgBounces');
        withHelp(this.probeBudget, 'lpgProbes');
        withHelp(this.vxgiBounces, 'vxgiBounces');
        const rebake = withHelp(h('button', { text: 'Bake again' }), 'lpgBake');
        rebake.addEventListener('click', () => this.handlers.onRebake());
        const restart = withHelp(h('button', { text: 'Start the probes over' }), 'ddgiRestart');
        restart.addEventListener('click', () => this.handlers.onDdgiRestart());

        this.probeColorRow.append(withHelp(h('div', { className: 'label', text: 'Colour the probes by' }), 'ddgiProbeColor'), this.probeColor);
        this.atlasEvRow.append(withHelp(h('div', { className: 'label' }, 'Atlas exposure ', this.atlasEvValue), 'ddgiAtlasExposure'), this.atlasEv);
        this.atlasOptions.append(withHelp(h('label', { className: 'inline' }, this.atlasFill, ' Fill the picture'), 'ddgiAtlasFill'), this.atlasEvRow);
        this.ddgiSettings.append(
            withHelp(h('div', { className: 'label' }, 'Update budget per frame ', this.budgetValue), 'ddgiBudget'),
            this.budget,
            withHelp(h('label', { className: 'inline' }, this.showProbes, ' Show the probes'), 'ddgiProbes'),
            this.probeColorRow,
            this.probeLegend,
            withHelp(h('div', { className: 'label', text: 'Debug view' }), 'ddgiScreenView'),
            this.screenView,
            withHelp(h('div', { className: 'label', text: 'Probe atlas over the picture' }), 'ddgiAtlas'),
            this.atlas,
            this.atlasOptions,
            withHelp(h('div', { className: 'label', text: 'Change per frame (flicker check)' }), 'ddgiChange'),
            this.change,
            restart,
        );
        this.probeSettings.append(
            h('div', { className: 'label', text: 'Bounces' }),
            this.probeBounces,
            h('div', { className: 'label', text: 'Probes' }),
            this.probeBudget,
            rebake,
        );
        this.vxgiSettings.append(h('div', { className: 'label', text: 'Bounces' }), this.vxgiBounces);
        withHelp(this.radianceCacheView, 'rcScreenView');
        this.radianceCacheSettings.append(withHelp(h('div', { className: 'label', text: 'Debug view' }), 'rcScreenView'), this.radianceCacheView);
        return h(
            'section',
            {},
            h('h2', { text: 'Global illumination' }),
            h(
                'div',
                { className: 'row' },
                radio('off', 'Off (direct)', 'giOff'),
                radio('on', 'On (both)', 'giOn'),
                radio('only', 'GI only (indirect)', 'giOnly'),
            ),
            h('div', { className: 'label', text: 'Technique' }),
            this.techniqueSelect,
            this.ddgiSettings,
            this.radianceCacheSettings,
            this.probeSettings,
            this.vxgiSettings,
            this.status,
        );
    }

    private compareSection(): HTMLElement {
        withHelp(this.wipeLeft, 'wipeLeft');
        withHelp(this.wipeRight, 'wipeRight');
        const reference = withHelp(h('button', { text: 'Compare DDGI with the Cycles reference' }), 'compareReference');
        reference.addEventListener('click', () => this.handlers.onCompareWithReference());
        withHelp(this.referenceNote, 'compareReference');
        return h(
            'section',
            {},
            h('h2', { text: 'Compare' }),
            reference,
            withHelp(h('label', { className: 'inline' }, this.wipeToggle, ' Split screen between two techniques'), 'wipeToggle'),
            h('div', { className: 'label', text: 'Left of the line' }),
            this.wipeLeft,
            h('div', { className: 'label', text: 'Right of the line' }),
            this.wipeRight,
            this.referenceNote,
        );
    }

    private upscalerSection(): HTMLElement {
        withHelp(this.upscaleQuality, 'upscaleQuality');
        return h(
            'section',
            {},
            h('h2', { text: 'Upscaler (optional)' }),
            withHelp(h('label', { className: 'inline' }, this.upscaleToggle, ' Draw smaller and upscale'), 'upscaleToggle'),
            this.upscaleQuality,
        );
    }

    private displaySection(): HTMLElement {
        const hide = withHelp(h('button', { text: 'Hide panel (H)' }), 'panelHide');
        hide.addEventListener('click', () => this.handlers.onPanelHidden(true));
        return h(
            'section',
            {},
            h('h2', { text: 'Display' }),
            withHelp(h('label', { className: 'inline' }, this.statsToggle, ' Show frame time, GPU time and DDGI stats'), 'statsToggle'),
            this.statsExplanation(),
            hide,
        );
    }

    /**
     * What each line of the stats box means. The box itself lets the mouse through, so dragging
     * over it always moves the camera; its explanations live here instead.
     */
    private statsExplanation(): HTMLElement {
        const entries = STATS_LINES.map(([name, key]) => h('div', { className: 'stat-explained' }, h('b', { text: name }), ` ${HELP[key]}`));
        return withHelp(h('details', { className: 'small' }, h('summary', { text: 'What the stats mean' }), ...entries), 'statsExplain');
    }

    // -- events ------------------------------------------------------------

    private ddgiPanelSettings(): DdgiPanelSettings {
        const view = this.screenView.value;
        const atlas = this.atlas.value;
        const change = this.change.value;
        return {
            budgetMs: Number(this.budget.value),
            probes: this.showProbes.checked,
            probeColor: this.probeColor.value as ProbeColorName,
            debug: view === 'none' ? null : (view as DdgiScreenViewName),
            atlas: atlas === 'none' ? null : (atlas as AtlasName),
            atlasFill: this.atlasFill.checked,
            atlasEv: Number(this.atlasEv.value),
            change: change === 'none' ? null : (Number(change) as ChangeFullScalePercent),
        };
    }

    private wireEvents(): void {
        const handlers = this.handlers;
        this.sceneSelect.addEventListener('change', () => handlers.onScene(this.sceneSelect.value));
        for (const [mode, input] of this.navButtons) input.addEventListener('change', () => input.checked && handlers.onNavigation(mode));
        // Fly speed is a log slider: -1 to 2.5 covers 0.1 to ~316 m/s.
        this.flySpeed.addEventListener('input', () => handlers.onFlySpeed(10 ** Number(this.flySpeed.value)));
        this.fov.addEventListener('input', () => handlers.onFieldOfView(Number(this.fov.value)));
        this.exposure.addEventListener('input', () => {
            this.showExposure(Number(this.exposure.value));
            handlers.onExposure(Number(this.exposure.value));
        });
        this.toneSelect.addEventListener('change', () => handlers.onToneMapping(this.toneSelect.value as ToneMappingName));
        this.referenceCube.addEventListener('change', () => handlers.onReferenceCube(this.referenceCube.checked));
        for (const [choice, input] of this.giButtons) input.addEventListener('change', () => input.checked && handlers.onGi(choice));
        this.techniqueSelect.addEventListener('change', () => handlers.onTechnique(this.techniqueSelect.value as GiTechniqueId));
        this.budget.addEventListener('input', () => {
            this.showBudget(Number(this.budget.value));
            handlers.onDdgiSettings(this.ddgiPanelSettings());
        });
        this.showProbes.addEventListener('change', () => handlers.onDdgiSettings(this.ddgiPanelSettings()));
        this.screenView.addEventListener('change', () => handlers.onDdgiSettings(this.ddgiPanelSettings()));
        for (const control of [this.probeColor, this.atlas, this.atlasFill, this.change]) {
            control.addEventListener('change', () => handlers.onDdgiSettings(this.ddgiPanelSettings()));
        }
        this.atlasEv.addEventListener('input', () => {
            this.showAtlasEv(Number(this.atlasEv.value));
            handlers.onDdgiSettings(this.ddgiPanelSettings());
        });
        const probeChanged = () =>
            handlers.onProbeSettings({ bounces: Number(this.probeBounces.value), maxProbes: Number(this.probeBudget.value) });
        this.probeBounces.addEventListener('change', probeChanged);
        this.probeBudget.addEventListener('change', probeChanged);
        this.vxgiBounces.addEventListener('change', () => handlers.onVxgiSettings({ bounces: Number(this.vxgiBounces.value) }));
        this.radianceCacheView.addEventListener('change', () => {
            const view = this.radianceCacheView.value;
            handlers.onRadianceCacheDebug(view === 'none' ? null : (view as RadianceCacheScreenViewName));
        });
        const wipeChanged = () =>
            handlers.onWipe(this.wipeToggle.checked, this.wipeLeft.value as TechniqueId, this.wipeRight.value as TechniqueId);
        this.wipeToggle.addEventListener('change', wipeChanged);
        this.wipeLeft.addEventListener('change', wipeChanged);
        this.wipeRight.addEventListener('change', wipeChanged);
        const upscaleChanged = () =>
            handlers.onUpscale(this.upscaleToggle.checked ? (this.upscaleQuality.value as UpscaleQuality) : null);
        this.upscaleToggle.addEventListener('change', upscaleChanged);
        this.upscaleQuality.addEventListener('change', upscaleChanged);
        this.statsToggle.addEventListener('change', () => handlers.onStats(this.statsToggle.checked));
    }

    private installHelp(): void {
        this.element.addEventListener('mouseover', (event) => {
            const target = (event.target as HTMLElement).closest<HTMLElement>('[data-help]');
            this.helpBox.textContent = target?.dataset.help ?? DEFAULT_HELP;
        });
        this.element.addEventListener('mouseleave', () => {
            this.helpBox.textContent = DEFAULT_HELP;
        });
    }

    // -- updating from the app ---------------------------------------------

    setHidden(hidden: boolean): void {
        this.element.classList.toggle('hidden', hidden);
    }

    get hidden(): boolean {
        return this.element.classList.contains('hidden');
    }

    setScene(manifest: SceneManifest): void {
        this.sceneSelect.value = manifest.id;
        this.sceneDescription.textContent = manifest.description;
        const credits = manifest.credits.filter((credit) => credit.asset !== null);
        this.sceneCredit.textContent = credits.length > 0 ? `Credit: ${credits.map((credit) => `${credit.license}, ${credit.attribution}`).join(' | ')}` : '';
        this.bookmarkRow.replaceChildren();
        this.bookmarkButtons = manifest.cameras.map((bookmark: CameraBookmark, index) => {
            const button = h('button', { text: `${index + 1}. ${bookmarkLabel(bookmark)}` });
            button.addEventListener('click', () => this.handlers.onBookmark(index));
            this.bookmarkRow.append(button);
            return button;
        });
    }

    setStatus(lines: readonly string[]): void {
        this.status.replaceChildren(...lines.map((line) => h('div', { text: line })));
    }

    /** The probe legend: one swatch per state with its count. Hidden unless the probes are shown. */
    setProbeLegend(entries: readonly ProbeLegendEntry[]): void {
        this.probeLegend.replaceChildren(
            ...entries.map((entry) =>
                h(
                    'div',
                    { className: 'legend-row' },
                    h('span', { className: 'swatch', style: `background: ${cssColor(entry.color)}` }),
                    h('span', { text: entry.label }),
                    h('span', { className: 'value', text: entry.count === null ? '' : entry.count.toLocaleString('en') }),
                ),
            ),
        );
    }

    /** A line under the compare controls about the reference image (what it is, or how to make it); null hides it. */
    setReferenceNote(text: string | null): void {
        this.referenceNote.hidden = text === null;
        this.referenceNote.textContent = text ?? '';
    }

    private showExposure(ev: number): void {
        const own = Math.abs(ev - this.sceneExposureEv) < 1e-6 && this.sceneExposureEv !== 0 ? " (this scene's own)" : '';
        this.exposureValue.textContent = `${formatEv(ev)}${own}`;
    }

    private showBudget(ms: number): void {
        this.budgetValue.textContent = `${ms.toFixed(2)} ms`;
    }

    private showAtlasEv(ev: number): void {
        this.atlasEvValue.textContent = formatEv(ev);
    }

    /** Shows `state` without firing any handler. */
    setState(state: PanelState): void {
        if (state.sceneId) this.sceneSelect.value = state.sceneId;
        this.bookmarkButtons.forEach((button, index) => button.classList.toggle('active', index === state.bookmarkIndex));
        this.navButtons.get(state.navigation)!.checked = true;
        this.flySpeed.value = String(Math.log10(Math.max(0.1, state.flySpeed)));
        this.fov.value = String(state.fieldOfView);
        this.sceneExposureEv = state.sceneExposureEv;
        this.exposure.value = String(state.exposureEv);
        this.showExposure(state.exposureEv);
        this.toneSelect.value = state.toneMapping;
        this.referenceCube.checked = state.referenceCube;
        this.giButtons.get(state.gi)!.checked = true;
        this.techniqueSelect.value = state.technique;
        this.budget.value = String(state.ddgi.budgetMs);
        this.showBudget(state.ddgi.budgetMs);
        this.showProbes.checked = state.ddgi.probes;
        this.screenView.value = state.ddgi.debug ?? 'none';
        this.probeColor.value = state.ddgi.probeColor;
        this.atlas.value = state.ddgi.atlas ?? 'none';
        this.atlasFill.checked = state.ddgi.atlasFill;
        this.atlasEv.value = String(state.ddgi.atlasEv);
        this.showAtlasEv(state.ddgi.atlasEv);
        this.change.value = state.ddgi.change === null ? 'none' : String(state.ddgi.change);
        this.probeBounces.value = String(state.probeBounces);
        this.probeBudget.value = String(state.maxProbes);
        this.vxgiBounces.value = String(state.vxgiBounces);
        this.radianceCacheView.value = state.radianceCacheDebug ?? 'none';
        this.wipeToggle.checked = state.wipe !== null;
        this.wipeLeft.value = state.wipe?.left ?? 'ddgi';
        this.wipeRight.value = state.wipe?.right ?? 'cycles';
        this.upscaleToggle.checked = state.upscale !== null;
        this.upscaleQuality.value = state.upscale ?? 'quality';
        this.statsToggle.checked = state.stats;
        // Show the settings of every technique on screen, and nothing for "GI off".
        const onScreen: TechniqueId[] = state.wipe ? [state.wipe.left, state.wipe.right] : [state.technique];
        const giShown = state.gi !== 'off';
        this.ddgiSettings.hidden = !(giShown && onScreen.includes('ddgi'));
        this.probeLegend.hidden = !state.ddgi.probes;
        this.probeColorRow.hidden = !state.ddgi.probes;
        this.atlasOptions.hidden = state.ddgi.atlas === null;
        // The exposure is the irradiance atlas's; distances are shown on a fixed scale.
        this.atlasEvRow.hidden = state.ddgi.atlas !== 'irradiance';
        this.probeSettings.hidden = !(giShown && onScreen.includes('lpg'));
        this.vxgiSettings.hidden = !(giShown && onScreen.includes('vxgi'));
        this.radianceCacheSettings.hidden = !(giShown && onScreen.includes('rc'));
        this.techniqueSelect.disabled = state.wipe !== null;
        this.wipeLeft.disabled = state.wipe === null;
        this.wipeRight.disabled = state.wipe === null;
        this.upscaleQuality.disabled = state.upscale === null || !state.upscaleAvailable;
        this.upscaleToggle.disabled = !state.upscaleAvailable;
    }
}
