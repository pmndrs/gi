/**
 * The viewer's moving-geometry scenarios (T3.8, docs/design/m3-moving.md §8):
 * each is a scene, a camera view and one of the scene's manifest motions,
 * picked from one list in the panel ("Moving things"), so anyone can watch
 * GI follow a moving object without knowing where things are.
 *
 * The motions live in the scene manifests (`scenes/manifests/*.json`), so the
 * same motion plays the same way in the viewer and in the checks. Each
 * scenario's step motion (a door that jumps shut, say) is what the response
 * meter measures.
 *
 * No engine imports, so it is unit-tested.
 */

export interface Scenario {
    /** Kebab-case, for links (`scenario=door`). */
    readonly id: string;
    /** The list's entry, in plain words. */
    readonly title: string;
    /** What it shows and what to watch for, in plain words: the panel shows it under the list. */
    readonly description: string;
    readonly scene: string;
    /** A camera view (bookmark) name of the scene. */
    readonly camera: string;
    /** The manifest motion that plays. */
    readonly motion: string;
    /** The motion the response meter offers for this scenario's step, if it has one. */
    readonly stepMotion: string | null;
    /** DDGI's camera-following volume on (the fly-through). */
    readonly followCamera: boolean;
    /** The panel offers "Add a box" and "Copy a box" in this scenario's scene. */
    readonly addsCabinets: boolean;
}

export const SCENARIOS: readonly Scenario[] = [
    {
        id: 'furniture',
        title: 'Furniture: a cabinet slides along the wall',
        description:
            "A red cabinet in the breakfast room slides 1 m along the wall, waits, and slides back. Watch its shadow on the floor and the reddish light it bounces onto the wall and floor: both follow it within about a second. The cabinet is only there while this plays; the room's own furniture stays put.",
        scene: 'furnished-room',
        camera: 'from-the-corner',
        motion: 'cabinet-slide',
        stepMotion: 'cabinet-step',
        followCamera: false,
        addsCabinets: false,
    },
    {
        id: 'door',
        title: 'Door: it swings shut between two rooms',
        description:
            "You are in the dark room, looking at the doorway. The lamp is in the other room. While the door is open, light spills through and brightens this room. The door swings shut in 1 s on the far side: this room should go dark within about a second and stay dark (no light leaking past the door), then light up again when the door opens.",
        scene: 'closed-door',
        camera: 'dark-room-leak-view',
        motion: 'door-swing',
        stepMotion: 'door-step',
        followCamera: false,
        addsCabinets: false,
    },
    {
        id: 'cabinets',
        title: 'Cabinets: a door opens; add and copy cabinets',
        description:
            "A kitchen run of hollow cabinets. The door of the second cabinet from the left swings open, so light gets into the cabinet, then closes again: inside should go dark. Below, \"Add a box\" puts a new small cabinet on the floor in front of the run (a new shape for GI), \"Copy a box\" adds one more of the same shape (cheaper for GI: it only places it again), up to 100.",
        scene: 'cabinet-row',
        camera: 'from-the-right',
        motion: 'cabinet-door-swing',
        stepMotion: 'cabinet-door-step',
        followCamera: false,
        addsCabinets: true,
    },
    {
        id: 'wind-tree',
        title: 'Wind: a tree sways in gusts',
        description:
            'A 10 m tree in a walled courtyard sways in gusts of wind; the slider below sets how strong the wind is. Its shadow moves on the ground and walls, and the bounced light under it follows. The leaves are solid cards until alpha cut-outs arrive (M4), so the shade under the tree is darker than it will be.',
        scene: 'courtyard-wind-tree',
        camera: 'from-the-south-east',
        motion: 'wind-gusts',
        stepMotion: 'wind-lean',
        followCamera: false,
        addsCabinets: false,
    },
    {
        id: 'fly-through',
        title: 'Fly-through: down the Bistro street',
        description:
            "The camera flies 66 m down the Bistro street at 5 m/s (a brisk run), turning twice, turns round and flies back the same way. DDGI adds a finer set of probes (1.5 m apart) that follows the camera, in front of the scene's own (4 m apart): watch for the bounced light staying steady as new probes come in ahead, with no popping or dark patches. Bistro takes a while to load the first time.",
        scene: 'bistro-exterior',
        camera: 'street',
        motion: 'fly-through',
        stepMotion: null,
        followCamera: true,
        addsCabinets: false,
    },
];

export function findScenario(id: string | null): Scenario | null {
    return SCENARIOS.find((scenario) => scenario.id === id) ?? null;
}

/** Whether a scene gets the cabinet scenario's "Add a box" and "Copy a box". */
export function sceneAddsCabinets(sceneId: string): boolean {
    return SCENARIOS.some((scenario) => scenario.scene === sceneId && scenario.addsCabinets);
}
