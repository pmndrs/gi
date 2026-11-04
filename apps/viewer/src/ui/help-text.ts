/**
 * Hover help for every control, in plain language: what it does and what you
 * will see. Kept in one file so the wording can be read and fixed in one go.
 */

export const HELP = {
    scene:
        'Pick a test scene. Each one opens on a lit view. Big ones (Sponza, Bistro) take a few seconds to load; a progress bar shows how far along it is.',
    cameraBookmarks:
        'Jump to a saved view of this scene. The same views are used by the native RTXGI tool, so pictures can be compared side by side.',
    copyLink: 'Copies a link that reopens exactly this view: scene, camera position, GI choice, exposure and split screen.',
    navOrbit: 'Orbit: drag to circle around what you are looking at, scroll to zoom, right-drag to slide sideways.',
    navFly: 'Fly: drag to look around, W A S D to move, Q and E (or C and Space) to go down and up, hold Shift to go faster, scroll to change speed.',
    flySpeed: 'How fast you move in fly mode, in metres per second. Scrolling in fly mode changes this too.',
    fieldOfView: 'How wide the camera sees. Smaller is zoomed in, larger is wide angle. Bookmarks set their own.',
    exposure:
        'Brightens or darkens the whole picture in photographic stops. 0 is the normal setting; +1 is twice as bright, -1 half. It does not change the lighting itself.',
    exposureReset: 'Back to 0, the normal exposure.',
    toneMapping:
        'How very bright light is squeezed onto the screen. ACES is the usual film look. AgX and Neutral keep colours truer. Linear clips bright areas to white.',
    referenceCube: 'Shows a 1 metre cube so you can judge the size of the scene.',
    giOff: 'Direct light only: the sun and lamps, plus shadows, but no light bouncing off walls. Shadowed areas go dark.',
    giOn: 'Direct light plus global illumination: light that has bounced around the scene, giving soft fill, colour bleeding and brighter shadows.',
    giOnly:
        'Only the bounced light, with the direct sun and lamps hidden. Use it to see what the GI technique alone contributes. Surfaces still show their colour, and glowing surfaces still glow.',
    technique:
        'Which global illumination technique to use. Both come from three.js and are the baselines our own technique has to beat. Light probe grid: light stored at points in the scene, baked once. VXGI: a voxel copy of the scene traced live every frame. Both get the same lights, sky and exposure, at full physical strength, so the comparison is fair.',
    lpgBounces:
        'How many times light bounces before it is stored in the probes. 0 is direct light only, 1 adds one bounce (the usual choice), 2 adds a second. More bounces take longer to bake.',
    lpgProbes: 'The most probes the grid may use. More probes capture finer lighting changes but bake slower. The spacing is widened to fit.',
    lpgBake: 'Throw the baked lighting away and bake again, with the current settings.',
    vxgiBounces:
        'How many bounces of light are cached in the voxel copy of the scene. 1 is the usual choice. VXGI only bounces the sun and lamps: it takes no light from the sky.',
    wipeToggle:
        'Split the screen between two techniques on the same view. The left side shows one, the right side the other, with a white line between them. Drag the line to compare.',
    wipeLeft: 'The technique drawn on the left of the white line. "No GI" is direct light only, for comparing against nothing.',
    wipeRight: 'The technique drawn on the right of the white line.',
    upscaleToggle:
        'Draws the picture at a lower resolution and rebuilds it to full size with @pmndrs/upscaler, which saves GPU time. Works with No GI and the light probe grid; VXGI has its own anti-aliasing, so this is switched off while VXGI is on screen.',
    upscaleQuality: 'How much smaller the picture is drawn. Quality is a little smaller and looks nearly sharp; Ultra performance is much smaller and softer.',
    statsToggle:
        'Shows the frame time, and the GPU time of one frame (averaged over the last few frames so it can be read). Needs a GPU that supports timestamps.',
    panelHide: 'Hides this panel so the picture fills the window. Press H, or the button at the top right, to bring it back.',
    wipeLine: 'Drag this line left or right to compare the two techniques.',
} as const;

export type HelpKey = keyof typeof HELP;

export const DEFAULT_HELP = 'Hover over any control to see what it does.';
