/**
 * The panel's "Moving things" section (T3.8): the scenario list, what the
 * chosen scenario shows, play and stop, a step measurement, and the controls
 * the open scene has: the wind strength for a swaying tree, and adding and
 * copying cabinets on the cabinet row. Like the other panel parts it only
 * builds controls and reports what the user did; `ViewerApp` decides what
 * that means. Every control has hover help (`help-text.ts`).
 */

import { MAX_WIND_REACH_M } from '../lighting/motion-drivers';
import { MAX_ADDED_CABINETS } from '../scene/added-cabinets';
import { SCENARIOS, type Scenario } from '../scene/scenarios';
import { h } from './dom';
import { withHelp } from './panel';

export interface ScenarioHandlers {
    /** null: "Choose a scenario..." again (nothing changes in the scene). */
    onScenario(id: string | null): void;
    /** Plays the chosen scenario's motion from the start, or stops it. */
    onPlayScenario(play: boolean): void;
    onMeasureScenario(): void;
    /** 0 to 1. */
    onWindStrength(strength: number): void;
    onAddCabinet(): void;
    onCopyCabinet(): void;
    onClearCabinets(): void;
}

export class ScenarioControls {
    readonly section = h('section');

    private readonly select = h('select');
    private readonly description = h('p', { className: 'muted small' });
    private readonly playRow = h('div', { className: 'row' });
    private readonly play = h('button', { text: 'Play' });
    private readonly measure = h('button', { text: 'Measure a step' });
    private readonly status = h('p', { className: 'muted small' });
    private readonly windRow = h('div');
    private readonly wind = h('input', { type: 'range', min: 0, max: 100, step: 5 });
    private readonly windValue = h('span', { className: 'value' });
    private readonly cabinetRows = h('div');
    private readonly addCabinet = h('button', { text: 'Add a box' });
    private readonly copyCabinet = h('button', { text: 'Copy a box' });
    private readonly clearCabinets = h('button', { text: 'Remove added' });
    private readonly cabinetStatus = h('p', { className: 'muted small' });

    private scenario: Scenario | null = null;

    constructor(private readonly handlers: ScenarioHandlers) {
        withHelp(this.select, 'scenarioSelect');
        withHelp(this.play, 'scenarioPlay');
        withHelp(this.measure, 'scenarioMeasure');
        withHelp(this.wind, 'windStrength');
        withHelp(this.addCabinet, 'cabinetAdd');
        withHelp(this.copyCabinet, 'cabinetCopy');
        withHelp(this.clearCabinets, 'cabinetClear');

        this.select.append(h('option', { value: '', text: 'Choose a scenario...' }));
        for (const scenario of SCENARIOS) this.select.append(h('option', { value: scenario.id, text: scenario.title }));
        this.select.addEventListener('change', () => this.handlers.onScenario(this.select.value === '' ? null : this.select.value));
        this.play.addEventListener('click', () => this.handlers.onPlayScenario(this.play.dataset.playing !== '1'));
        this.measure.addEventListener('click', () => this.handlers.onMeasureScenario());
        this.wind.addEventListener('input', () => {
            this.showWind(Number(this.wind.value) / 100);
            this.handlers.onWindStrength(Number(this.wind.value) / 100);
        });
        this.addCabinet.addEventListener('click', () => this.handlers.onAddCabinet());
        this.copyCabinet.addEventListener('click', () => this.handlers.onCopyCabinet());
        this.clearCabinets.addEventListener('click', () => this.handlers.onClearCabinets());

        this.playRow.append(this.play, this.measure);
        this.windRow.append(withHelp(h('div', { className: 'label' }, 'Wind strength ', this.windValue), 'windStrength'), this.wind);
        this.cabinetRows.append(h('div', { className: 'row' }, this.addCabinet, this.copyCabinet, this.clearCabinets), this.cabinetStatus);
        this.section.append(
            h('h2', { text: 'Moving things' }),
            withHelp(h('div', { className: 'label', text: 'Scenario' }), 'scenarioSelect'),
            this.select,
            this.description,
            this.playRow,
            this.status,
            this.windRow,
            this.cabinetRows,
        );
        this.setScenario(null, false);
        this.setSceneControls({ wind: false, cabinets: false });
    }

    /** The chosen scenario (null: none) and whether it can be played in the open scene. */
    setScenario(scenario: Scenario | null, available: boolean): void {
        this.scenario = scenario;
        this.select.value = scenario?.id ?? '';
        this.description.textContent = scenario ? scenario.description : 'Each scenario opens a scene and moves something in it, so you can watch GI follow.';
        this.playRow.hidden = scenario === null || !available;
        this.measure.hidden = scenario?.stepMotion === null;
    }

    /** Whether the scenario's motion is playing now, with a line about it. */
    setPlaying(playing: boolean, status: string): void {
        this.play.textContent = playing ? 'Stop' : 'Play';
        this.play.dataset.playing = playing ? '1' : '0';
        this.status.textContent = status;
    }

    /** The controls the open scene has. */
    setSceneControls(controls: { wind: boolean; cabinets: boolean }): void {
        this.windRow.hidden = !controls.wind;
        this.cabinetRows.hidden = !controls.cabinets;
    }

    setWindStrength(strength: number): void {
        this.wind.value = String(Math.round(strength * 100));
        this.showWind(strength);
    }

    setCabinetCount(count: number, shapes: number): void {
        this.cabinetStatus.textContent =
            count === 0 ? `None added yet (up to ${MAX_ADDED_CABINETS}).` : `${count} added: ${shapes} with a shape of their own, ${count - shapes} copies (up to ${MAX_ADDED_CABINETS}).`;
        this.addCabinet.disabled = count >= MAX_ADDED_CABINETS;
        this.copyCabinet.disabled = count >= MAX_ADDED_CABINETS;
        this.clearCabinets.disabled = count === 0;
    }

    private showWind(strength: number): void {
        this.windValue.textContent = `${Math.round(strength * 100)}% (tree top moves up to ${(strength * MAX_WIND_REACH_M).toFixed(1)} m)`;
    }
}
