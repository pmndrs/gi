/**
 * The panel's controls for changing light while DDGI runs (T2.8,
 * docs/design/m2-lights.md §7): the scene's lights (with a gizmo in the
 * picture), its scripted motions, time of day, light groups and the
 * response meter. Like `panel.ts`, it only builds controls and reports what
 * the user did; `ViewerApp` decides what that means. Every control has
 * hover help (`help-text.ts`).
 */

import { Color } from 'three';
import type { ManifestMotion } from '../../../../scenes/src/manifest';
import type { LightKind, LightSettings } from '../lighting/light-editor';
import type { MeterResult } from '../lighting/response-meter';
import { CHANGE_SHARE, TOLERANCE } from '../stats/response-curve';
import type { GroupWeight, LightGroupCount } from '../techniques/ddgi-technique';
import { h } from './dom';
import { withHelp } from './panel';

export interface LightingHandlers {
    onSelectLight(index: number | null): void;
    onLightSettings(settings: LightSettings): void;
    onPutBack(): void;
    onGizmoMode(mode: 'translate' | 'rotate'): void;
    /** null stops the motion playing. */
    onPlayMotion(motionId: string | null): void;
    onTimeOfDay(enabled: boolean): void;
    onTime(hour: number, dayOfYear: number): void;
    onPlayDay(): void;
    onLightGroups(count: LightGroupCount): void;
    onGroupWeight(group: number | 'environment', weight: GroupWeight): void;
    onPickRegion(): void;
    onWholePicture(): void;
    /** One jump of a motion: from keyframe `jump` to keyframe `jump + 1`. */
    onMeasureStep(motionId: string, jump: number): void;
    onCancelMeasure(): void;
    onSaveCsv(): void;
}

/** A light as the list shows it. */
export interface LightEntry {
    label: string;
    kind: LightKind;
}

/** Brightness slider: stops from the light's own brightness, each notch a stop. */
const MIN_STOPS = -10;
const MAX_STOPS = 4;
/** Light group weight sliders run from off to twice as bright. */
const MAX_GROUP_WEIGHT = 2;
const KIND_LABELS: Record<LightKind, string> = { point: 'lamp', spot: 'spotlight', directional: 'sun' };
const UNITS: Record<LightKind, string> = { point: 'cd', spot: 'cd', directional: 'lux' };
const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

/** "21 June" for day 172. */
export function dayLabel(dayOfYear: number): string {
    let day = Math.round(dayOfYear);
    for (let month = 0; month < 12; month++) {
        if (day <= MONTH_DAYS[month]) return `${day} ${MONTHS[month]}`;
        day -= MONTH_DAYS[month];
    }
    return `31 December`;
}

/** "14:30" for 14.5. */
export function hourLabel(hour: number): string {
    const minutes = Math.round(hour * 60) % (24 * 60);
    return `${String(Math.floor(minutes / 60)).padStart(2, '0')}:${String(minutes % 60).padStart(2, '0')}`;
}

/** A linear colour as the colour picker's sRGB hex. */
function hexOf(color: readonly [number, number, number]): string {
    return `#${new Color().setRGB(color[0], color[1], color[2]).getHexString()}`;
}

/** The colour picker's sRGB hex as a linear colour (three converts it to its linear working space). */
function linearOf(hex: string): [number, number, number] {
    const color = new Color(hex);
    return [color.r, color.g, color.b];
}

function formatNumber(value: number): string {
    if (value === 0) return '0';
    if (value >= 100) return value.toFixed(0);
    if (value >= 10) return value.toFixed(1);
    return value.toPrecision(2);
}

export class LightingControls {
    readonly sections: HTMLElement[];

    // Lights
    private readonly lightsSection = h('section');
    private readonly lightSelect = h('select');
    private readonly lightBody = h('div');
    private readonly lightOn = h('input', { type: 'checkbox' });
    private readonly intensity = h('input', { type: 'range', min: MIN_STOPS, max: MAX_STOPS, step: 0.25 });
    private readonly intensityValue = h('span', { className: 'value' });
    private readonly color = h('input', { type: 'color' });
    private readonly rangeRow = h('div');
    private readonly range = h('input', { type: 'range', min: 0, max: 30, step: 0.1 });
    private readonly rangeValue = h('span', { className: 'value' });
    private readonly spotRows = h('div');
    private readonly cone = h('input', { type: 'range', min: 1, max: 89, step: 1 });
    private readonly coneValue = h('span', { className: 'value' });
    private readonly penumbra = h('input', { type: 'range', min: 0, max: 1, step: 0.05 });
    private readonly penumbraValue = h('span', { className: 'value' });
    private readonly shadows = h('input', { type: 'checkbox' });
    private readonly group = h('select');
    private readonly gizmoRow = h('div', { className: 'row' });
    private readonly moveButton = h('button', { text: 'Move' });
    private readonly turnButton = h('button', { text: 'Turn' });

    // Motions
    private readonly motionsSection = h('section');
    private readonly motionSelect = h('select');
    private readonly motionPlay = h('button', { text: 'Play' });
    private readonly motionStatus = h('p', { className: 'muted small' });

    // Time of day
    private readonly timeSection = h('section');
    private readonly timeToggle = h('input', { type: 'checkbox' });
    private readonly hour = h('input', { type: 'range', min: 0, max: 24, step: 0.05 });
    private readonly hourValue = h('span', { className: 'value' });
    private readonly day = h('input', { type: 'range', min: 1, max: 365, step: 1 });
    private readonly dayValue = h('span', { className: 'value' });
    private readonly playDay = h('button', { text: 'Play a 10-second day' });

    // Light groups
    private readonly groupsSection = h('section');
    private readonly groupCount = h('select');
    private readonly groupRows = h('div');
    private groupControls: Array<{ key: number | 'environment'; color: HTMLInputElement; weight: HTMLInputElement; value: HTMLElement }> = [];

    // Response meter
    private readonly meterSection = h('section');
    private readonly meterMotion = h('select');
    private readonly meterMeasure = h('button', { text: 'Measure a step' });
    private readonly meterStatus = h('p', { className: 'muted small' });
    private readonly meterRegion = h('p', { className: 'muted small' });
    private readonly meterPlot = h('canvas', { width: 290, height: 130, className: 'meter-plot' });
    private readonly meterMarks = h('p', { className: 'small' });
    private readonly meterSave = h('button', { text: 'Save the curve as CSV' });

    private lightKind: LightKind | null = null;
    /** The chosen light's own brightness: the brightness slider counts stops from it. */
    private baseIntensity = 1;
    private measuring = false;

    constructor(private readonly handlers: LightingHandlers) {
        this.sections = [this.buildLights(), this.buildMotions(), this.buildTimeOfDay(), this.buildGroups(), this.buildMeter()];
    }

    // -- building ----------------------------------------------------------

    private buildLights(): HTMLElement {
        withHelp(this.lightSelect, 'lightSelect');
        withHelp(this.intensity, 'lightIntensity');
        withHelp(this.color, 'lightColor');
        withHelp(this.range, 'lightRange');
        withHelp(this.cone, 'lightCone');
        withHelp(this.penumbra, 'lightPenumbra');
        withHelp(this.group, 'lightGroup');
        withHelp(this.moveButton, 'lightGizmoMove');
        withHelp(this.turnButton, 'lightGizmoTurn');
        for (let group = 0; group < 4; group++) this.group.append(h('option', { value: group, text: `Group ${group}` }));
        const putBack = withHelp(h('button', { text: 'Put this light back' }), 'lightPutBack');
        putBack.addEventListener('click', () => this.handlers.onPutBack());
        this.gizmoRow.append(this.moveButton, this.turnButton);
        this.rangeRow.append(withHelp(h('div', { className: 'label' }, 'Reach ', this.rangeValue), 'lightRange'), this.range);
        this.spotRows.append(
            withHelp(h('div', { className: 'label' }, 'Cone ', this.coneValue), 'lightCone'),
            this.cone,
            withHelp(h('div', { className: 'label' }, 'Soft edge ', this.penumbraValue), 'lightPenumbra'),
            this.penumbra,
        );
        this.lightBody.append(
            withHelp(h('label', { className: 'inline' }, this.lightOn, ' On'), 'lightOn'),
            withHelp(h('div', { className: 'label' }, 'Brightness ', this.intensityValue), 'lightIntensity'),
            this.intensity,
            withHelp(h('div', { className: 'label', text: 'Colour' }), 'lightColor'),
            this.color,
            this.rangeRow,
            this.spotRows,
            withHelp(h('label', { className: 'inline' }, this.shadows, ' Casts shadows'), 'lightShadows'),
            withHelp(h('div', { className: 'label', text: 'Light group (for GI)' }), 'lightGroup'),
            this.group,
            this.gizmoRow,
            putBack,
        );
        const changed = () => this.handlers.onLightSettings(this.readLight());
        this.lightSelect.addEventListener('change', () => this.handlers.onSelectLight(this.lightSelect.value === 'none' ? null : Number(this.lightSelect.value)));
        this.intensity.addEventListener('input', () => {
            this.showIntensity();
            changed();
        });
        for (const control of [this.lightOn, this.shadows, this.group]) control.addEventListener('change', changed);
        this.color.addEventListener('input', changed);
        this.range.addEventListener('input', () => {
            this.rangeValue.textContent = this.rangeText(Number(this.range.value));
            changed();
        });
        this.cone.addEventListener('input', () => {
            this.coneValue.textContent = `${this.cone.value}°`;
            changed();
        });
        this.penumbra.addEventListener('input', () => {
            this.penumbraValue.textContent = Number(this.penumbra.value).toFixed(2);
            changed();
        });
        this.moveButton.addEventListener('click', () => this.handlers.onGizmoMode('translate'));
        this.turnButton.addEventListener('click', () => this.handlers.onGizmoMode('rotate'));
        this.lightsSection.append(h('h2', { text: 'Lights' }), this.lightSelect, this.lightBody);
        return this.lightsSection;
    }

    private buildMotions(): HTMLElement {
        withHelp(this.motionSelect, 'motionSelect');
        withHelp(this.motionPlay, 'motionPlay');
        this.motionPlay.addEventListener('click', () => this.handlers.onPlayMotion(this.motionPlay.dataset.playing === '1' ? null : this.motionSelect.value));
        this.motionsSection.append(h('h2', { text: 'Light motions' }), this.motionSelect, this.motionPlay, this.motionStatus);
        return this.motionsSection;
    }

    private buildTimeOfDay(): HTMLElement {
        withHelp(this.hour, 'timeOfDayHour');
        withHelp(this.day, 'timeOfDayDay');
        withHelp(this.playDay, 'timeOfDayPlay');
        this.timeToggle.addEventListener('change', () => this.handlers.onTimeOfDay(this.timeToggle.checked));
        const timeChanged = () => {
            this.hourValue.textContent = hourLabel(Number(this.hour.value));
            this.dayValue.textContent = dayLabel(Number(this.day.value));
            this.handlers.onTime(Number(this.hour.value), Number(this.day.value));
        };
        this.hour.addEventListener('input', timeChanged);
        this.day.addEventListener('input', timeChanged);
        this.playDay.addEventListener('click', () => this.handlers.onPlayDay());
        this.timeSection.append(
            h('h2', { text: 'Time of day' }),
            withHelp(h('label', { className: 'inline' }, this.timeToggle, ' Move the sun'), 'timeOfDayToggle'),
            withHelp(h('div', { className: 'label' }, 'Time ', this.hourValue), 'timeOfDayHour'),
            this.hour,
            withHelp(h('div', { className: 'label' }, 'Day ', this.dayValue), 'timeOfDayDay'),
            this.day,
            this.playDay,
        );
        return this.timeSection;
    }

    private buildGroups(): HTMLElement {
        withHelp(this.groupCount, 'lightGroupsCount');
        for (const [count, label] of [
            [0, 'Off (one layer for all light)'],
            [2, '2 groups, plus the sky'],
            [4, '4 groups, plus the sky'],
        ] as const) {
            this.groupCount.append(h('option', { value: count, text: label }));
        }
        this.groupCount.addEventListener('change', () => this.handlers.onLightGroups(Number(this.groupCount.value) as LightGroupCount));
        this.groupsSection.append(h('h2', { text: 'Light groups (GI)' }), withHelp(h('div', { className: 'label', text: 'Separate layers' }), 'lightGroupsCount'), this.groupCount, this.groupRows);
        return this.groupsSection;
    }

    private buildMeter(): HTMLElement {
        withHelp(this.meterMotion, 'meterMeasure');
        withHelp(this.meterMeasure, 'meterMeasure');
        withHelp(this.meterPlot, 'meterPlot');
        withHelp(this.meterMarks, 'meterPlot');
        withHelp(this.meterSave, 'meterSaveCsv');
        const pick = withHelp(h('button', { text: 'Pick a region' }), 'meterPickRegion');
        const whole = withHelp(h('button', { text: 'Whole picture' }), 'meterWholePicture');
        pick.addEventListener('click', () => this.handlers.onPickRegion());
        whole.addEventListener('click', () => this.handlers.onWholePicture());
        this.meterMeasure.addEventListener('click', () => {
            if (this.measuring) return this.handlers.onCancelMeasure();
            const [motionId, jump] = this.meterMotion.value.split('#');
            this.handlers.onMeasureStep(motionId, Number(jump));
        });
        this.meterSave.addEventListener('click', () => this.handlers.onSaveCsv());
        this.meterSave.disabled = true;
        this.meterSection.append(
            h('h2', { text: 'Response meter' }),
            h('div', { className: 'row' }, pick, whole),
            this.meterRegion,
            withHelp(h('div', { className: 'label', text: 'Step to measure' }), 'meterMeasure'),
            this.meterMotion,
            this.meterMeasure,
            this.meterStatus,
            this.meterPlot,
            this.meterMarks,
            this.meterSave,
        );
        return this.meterSection;
    }

    // -- reading -----------------------------------------------------------

    private readLight(): LightSettings {
        return {
            on: this.lightOn.checked,
            intensity: this.baseIntensity * 2 ** Number(this.intensity.value),
            color: linearOf(this.color.value),
            range: Number(this.range.value),
            coneDeg: Number(this.cone.value),
            penumbra: Number(this.penumbra.value),
            shadows: this.shadows.checked,
            group: Number(this.group.value),
        };
    }

    private showIntensity(): void {
        const unit = this.lightKind ? UNITS[this.lightKind] : '';
        const value = this.baseIntensity * 2 ** Number(this.intensity.value);
        const own = Math.abs(Number(this.intensity.value)) < 1e-6 ? " (the scene's own)" : '';
        this.intensityValue.textContent = `${formatNumber(value)} ${unit}${own}`;
    }

    private rangeText(metres: number): string {
        return metres <= 0 ? 'no limit' : `${metres.toFixed(1)} m`;
    }

    // -- updating from the app ---------------------------------------------

    /** A new scene: its lights, its motions, and whether it has a sun to move. */
    setScene(lights: readonly LightEntry[], motions: readonly ManifestMotion[], timeOfDay: { available: boolean; hour: number; dayOfYear: number }, rangeMax: number): void {
        this.lightSelect.replaceChildren(h('option', { value: 'none', text: lights.length > 0 ? 'Choose a light...' : 'This scene has no lights to change' }));
        lights.forEach((light, index) => this.lightSelect.append(h('option', { value: index, text: `${light.label} (${KIND_LABELS[light.kind]})` })));
        this.lightsSection.hidden = lights.length === 0;
        this.range.max = String(Math.max(10, Math.ceil(rangeMax)));
        this.setLight(null, null, 1);

        this.motionSelect.replaceChildren(...motions.map((motion) => h('option', { value: motion.id, text: motion.label })));
        this.motionsSection.hidden = motions.length === 0;
        this.setMotionPlaying(null, '');
        // Each jump of a motion is a step to measure, from one keyframe to the next; step motions first.
        const steps = [...motions].sort((a, b) => Number(b.interpolation === 'step') - Number(a.interpolation === 'step'));
        this.meterMotion.replaceChildren(
            ...steps.flatMap((motion) =>
                motion.keyframes.slice(1).map((keyframe, jump) =>
                    h('option', { value: `${motion.id}#${jump}`, text: `${motion.label}: jump ${jump + 1} (${motion.keyframes[jump].t} s to ${keyframe.t} s)` }),
                ),
            ),
        );
        this.meterMeasure.disabled = steps.length === 0;
        this.meterStatus.textContent = steps.length === 0 ? 'This scene has no light motion to make a step with.' : '';

        this.timeSection.hidden = !timeOfDay.available;
        this.timeToggle.checked = false;
        this.setTime(timeOfDay.hour, timeOfDay.dayOfYear);
        this.setMeterResult(null, []);
    }

    /** Shows the chosen light (null: none) and its settings; `ownIntensity` is the brightness the slider counts stops from. */
    setLight(index: number | null, light: { kind: LightKind; settings: LightSettings } | null, ownIntensity: number): void {
        this.lightSelect.value = index === null ? 'none' : String(index);
        this.lightBody.hidden = light === null;
        if (!light) return;
        const { kind, settings } = light;
        this.lightKind = kind;
        this.baseIntensity = ownIntensity > 0 ? ownIntensity : 1;
        this.lightOn.checked = settings.on;
        const stops = Math.log2(Math.max(settings.intensity, 1e-6) / this.baseIntensity);
        this.intensity.value = String(Math.min(MAX_STOPS, Math.max(MIN_STOPS, stops)));
        this.showIntensity();
        this.color.value = hexOf(settings.color);
        this.rangeRow.hidden = kind === 'directional';
        this.range.value = String(settings.range);
        this.rangeValue.textContent = this.rangeText(settings.range);
        this.spotRows.hidden = kind !== 'spot';
        this.cone.value = String(Math.round(settings.coneDeg));
        this.coneValue.textContent = `${Math.round(settings.coneDeg)}°`;
        this.penumbra.value = String(settings.penumbra);
        this.penumbraValue.textContent = settings.penumbra.toFixed(2);
        this.shadows.checked = settings.shadows;
        this.group.value = String(settings.group);
        this.gizmoRow.hidden = kind === 'directional';
        this.turnButton.hidden = kind !== 'spot';
    }

    setGizmoMode(mode: 'translate' | 'rotate'): void {
        this.moveButton.classList.toggle('active', mode === 'translate');
        this.turnButton.classList.toggle('active', mode === 'rotate');
    }

    setMotionPlaying(motionId: string | null, status: string): void {
        if (motionId) this.motionSelect.value = motionId;
        this.motionPlay.textContent = motionId ? 'Stop' : 'Play';
        this.motionPlay.dataset.playing = motionId ? '1' : '0';
        this.motionStatus.textContent = status;
    }

    setTimeOfDayEnabled(enabled: boolean, playing: boolean): void {
        this.timeToggle.checked = enabled;
        this.playDay.textContent = playing ? 'Stop the day' : 'Play a 10-second day';
    }

    setTime(hour: number, dayOfYear: number): void {
        this.hour.value = String(hour);
        this.day.value = String(Math.round(dayOfYear));
        this.hourValue.textContent = hourLabel(hour);
        this.dayValue.textContent = dayLabel(dayOfYear);
    }

    /** Light groups: the count DDGI runs with, and each layer's weight; `available` is false while DDGI isn't on screen. */
    setGroups(count: LightGroupCount, weights: readonly GroupWeight[], environment: GroupWeight, available: boolean): void {
        this.groupsSection.hidden = !available;
        this.groupCount.value = String(count);
        const keys: Array<number | 'environment'> = count === 0 ? [] : [...Array.from({ length: count }, (_, group) => group), 'environment'];
        const same = keys.length === this.groupControls.length && keys.every((key, index) => this.groupControls[index].key === key);
        if (!same) {
            this.groupControls = keys.map((key) => {
                const color = withHelp(h('input', { type: 'color' }), key === 'environment' ? 'environmentWeight' : 'lightGroupWeight');
                const weight = withHelp(h('input', { type: 'range', min: 0, max: MAX_GROUP_WEIGHT, step: 0.05 }), key === 'environment' ? 'environmentWeight' : 'lightGroupWeight');
                const value = h('span', { className: 'value' });
                const changed = () => {
                    value.textContent = `x${Number(weight.value).toFixed(2)}`;
                    this.handlers.onGroupWeight(key, { color: linearOf(color.value), intensity: Number(weight.value) });
                };
                color.addEventListener('input', changed);
                weight.addEventListener('input', changed);
                return { key, color, weight, value };
            });
            this.groupRows.replaceChildren(
                ...this.groupControls.flatMap((control) => [
                    withHelp(h('div', { className: 'label' }, control.key === 'environment' ? 'Sky and glowing surfaces ' : `Group ${control.key} `, control.value), control.key === 'environment' ? 'environmentWeight' : 'lightGroupWeight'),
                    h('div', { className: 'row' }, control.color, control.weight),
                ]),
            );
        }
        for (const control of this.groupControls) {
            const weight = control.key === 'environment' ? environment : (weights[control.key] ?? { color: [1, 1, 1], intensity: 1 });
            control.color.value = hexOf(weight.color);
            control.weight.value = String(weight.intensity);
            control.value.textContent = `x${weight.intensity.toFixed(2)}`;
        }
    }

    /** The meter's region in words. */
    setMeterRegion(text: string): void {
        this.meterRegion.textContent = text;
    }

    setMeterStatus(text: string, measuring: boolean): void {
        this.measuring = measuring;
        this.meterStatus.textContent = text;
        this.meterMeasure.textContent = measuring ? 'Stop measuring' : 'Measure a step';
        withHelp(this.meterMeasure, measuring ? 'meterCancel' : 'meterMeasure');
    }

    /** Draws the curve so far (`live`) or a finished result, with its marks. */
    setMeterResult(result: MeterResult | null, live: ReadonlyArray<{ frame: number; mean: number }>): void {
        this.meterSave.disabled = result === null;
        const points = result ? result.samples.map((sample) => ({ frame: sample.frame, mean: sample.mean })) : [...live].sort((a, b) => a.frame - b.frame);
        drawPlot(this.meterPlot, points, result);
        if (!result) {
            this.meterMarks.textContent = '';
            return;
        }
        const marks = result.marks;
        const frame = (value: number | null) => (value === null ? 'never' : `frame ${value}`);
        const forGood = marks.settledFrame !== marks.enteredFrame ? `, and for good from ${frame(marks.settledFrame)}` : '';
        this.meterMarks.textContent =
            `Step ${(marks.stepShare * 100).toFixed(0)}% of the picture's brightness. ` +
            `First changed by more than ${CHANGE_SHARE * 100}%: ${frame(marks.firstChangedFrame)}. ` +
            `Within ${TOLERANCE * 100}% of settled: ${frame(marks.enteredFrame)}${forGood}.` +
            (marks.overshootShareOfStep > 0.005 ? ` It went past settled by ${(marks.overshootShareOfStep * 100).toFixed(0)}% of the step.` : '');
    }
}

/** The meter's plot: before (grey dashed), settled (white) with its ±5% band, the curve, and the two marks. */
function drawPlot(canvas: HTMLCanvasElement, points: ReadonlyArray<{ frame: number; mean: number }>, result: MeterResult | null): void {
    const context = canvas.getContext('2d');
    if (!context) return;
    const { width, height } = canvas;
    context.clearRect(0, 0, width, height);
    context.fillStyle = '#0d1117';
    context.fillRect(0, 0, width, height);
    if (points.length === 0) return;
    const values = points.map((point) => point.mean);
    if (result) values.push(result.marks.before, result.marks.settled * (1 + TOLERANCE), result.marks.settled * (1 - TOLERANCE));
    let low = Math.min(...values);
    let high = Math.max(...values);
    if (high - low < 1e-9) {
        high += 1e-6;
        low -= 1e-6;
    }
    const lastFrame = Math.max(...points.map((point) => point.frame), 1);
    const margin = 6;
    const x = (frame: number) => margin + ((frame - 1) / Math.max(1, lastFrame - 1)) * (width - 2 * margin);
    const y = (value: number) => height - margin - ((value - low) / (high - low)) * (height - 2 * margin);
    if (result) {
        const { before, settled } = result.marks;
        context.fillStyle = 'rgba(255, 255, 255, 0.10)';
        const top = y(settled * (1 + TOLERANCE));
        context.fillRect(0, top, width, y(settled * (1 - TOLERANCE)) - top);
        context.strokeStyle = '#ffffff';
        context.beginPath();
        context.moveTo(0, y(settled));
        context.lineTo(width, y(settled));
        context.stroke();
        context.setLineDash([4, 4]);
        context.strokeStyle = '#8b96a3';
        context.beginPath();
        context.moveTo(0, y(before));
        context.lineTo(width, y(before));
        context.stroke();
        context.setLineDash([]);
        const markLine = (frame: number | null, color: string) => {
            if (frame === null) return;
            context.strokeStyle = color;
            context.beginPath();
            context.moveTo(x(frame), 0);
            context.lineTo(x(frame), height);
            context.stroke();
        };
        markLine(result.marks.firstChangedFrame, '#ff9d3c');
        markLine(result.marks.enteredFrame, '#4fd17a');
    }
    context.strokeStyle = '#5db0ff';
    context.lineWidth = 1.5;
    context.beginPath();
    points.forEach((point, index) => (index === 0 ? context.moveTo(x(point.frame), y(point.mean)) : context.lineTo(x(point.frame), y(point.mean))));
    context.stroke();
    context.lineWidth = 1;
}
