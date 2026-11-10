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
        'Brightens or darkens the whole picture in photographic stops: +1 is twice as bright, -1 half. Each scene opens at its own setting (Sponza +2, because its sun is weak and most of it is in shade). It does not change the lighting itself.',
    exposureReset: "Back to this scene's own exposure.",
    toneMapping:
        'How very bright light is squeezed onto the screen. ACES is the usual film look. AgX and Neutral keep colours truer. Linear clips bright areas to white.',
    referenceCube: 'Shows a 1 metre cube so you can judge the size of the scene.',
    giOff: 'Direct light only: the sun and lamps, plus shadows, but no light bouncing off walls. Shadowed areas go dark. Nothing here comes from DDGI, so this side never flickers.',
    giOn: 'Direct light plus global illumination (the indirect light): light that has bounced around the scene, giving soft fill, colour bleeding and brighter shadows.',
    giOnly:
        "Only the indirect light (the GI), with the direct sun and lamps hidden. Use it to see what the GI technique alone contributes, and where DDGI's noise is: without the steady direct light on top, any flicker is easier to see. Surfaces still show their colour, and glowing surfaces still glow.",
    technique:
        'Which global illumination technique to use. DDGI is ours: probes in a grid, each tracing rays every frame, so light, sky and moving things update live. The other two come from three.js and are the baselines DDGI has to beat. Light probe grid: light stored at points in the scene, baked once. VXGI: a voxel copy of the scene traced live every frame. All get the same lights, sky and exposure, at full physical strength, so the comparison is fair.',
    ddgiPreset:
        "A ready-made set of DDGI settings. High is the default: 256 rays per probe, the finest probe tiles, 2 ms a frame and full-size textures for GI. Medium and Low trace fewer rays per probe (192 and 128), keep coarser tiles for what each probe can see (so thin walls and small gaps blur a little more), start with a smaller update budget (1.5 and 1 ms; the slider below moves to it), and Low keeps GI's copy of the textures at a quarter of the size. Low is meant for laptops without a graphics card, tablets and phones. Changing it starts DDGI over (a few seconds): rays and tile sizes decide how big DDGI's buffers are, so they can only change at a fresh start. The probe spacing stays the scene's own.",
    ddgiBudget:
        'How much GPU time DDGI may spend per frame updating its probes, in milliseconds, with the GPU working at full speed. DDGI measures once how many probes that buys and then updates that many every frame; changing this works out the new number from what it measured. More updates more probes each frame, so light settles faster after a change; less leaves more time for drawing. 2 ms is the default and the target. The stats box (top left) shows how many probes that buys.',
    ddgiProbes:
        'Draws every DDGI probe as a small ball where it really sits (probes stuck in walls are moved out), coloured by what it is doing. The legend below says what each colour means; the numbers count the probes in each state.',
    ddgiProbeColor:
        "What the probe balls' colour shows. State: active, switched off, or not updated yet. How settled: how many updates each probe's light is averaged over, red for a few (a new or rarely updated probe, noisy), yellow for about 13, green for the most DDGI keeps (200, as quiet as it gets). Probes that are not active are grey. A probe updated every frame turns green within a few seconds; in a big scene each probe is updated less often, and DDGI lets older updates fade after a few seconds so light changes still come through, so those probes stay orange or yellow on purpose.",
    ddgiGiTextures:
        "Whether GI's rays read the materials' textures where they hit (on by default). Off, every texture counts as its average colour, as before textures reached GI: compare the two to see what the textures change. Turning it over takes a few seconds for the light to follow.",
    ddgiScreenView:
        "Replaces the picture with one of DDGI's check views. GI only: just the bounced light, as it would fall on plain white surfaces. Coverage: green where DDGI answers fully, red where it gives way (outside the probe box, or where no probe can see). Probes near this point: the colours of the 8 probes each point reads from. GI albedo at hits: the colour GI's rays find where they hit each surface (its texture, cutouts and metals as GI reads them; glass GI sees through shows what is behind it; black where a ray hits nothing), with no light. Turns off while the upscaler is on.",
    ddgiAtlas:
        "Draws one of DDGI's probe textures over the picture. Each small square is one probe: the light it has gathered from every direction (irradiance), or how far it can see in every direction (distance). Each block of squares is one horizontal layer of probes, the lowest first (top left). A square that changes from frame to frame is a noisy probe; a black one is switched off or not updated yet. Irradiance is shown as the light on a white wall facing that way, at the picture's exposure plus the atlas exposure below; distance goes from black (right at the probe) to white (one and a half probe cells away or more).",
    ddgiAtlasFill: 'Makes the probe texture fill the picture instead of sitting in the bottom-right corner, so single squares are easier to see. It stays below the stats box; hide the stats (Display) to give it the whole picture.',
    ddgiAtlasExposure:
        'Brightens or darkens the irradiance texture only, in stops, on top of the picture\'s exposure. Dark probes become readable at +2 to +4. It does not affect the distance texture.',
    ddgiChange:
        "Shows how much each pixel changed since the previous frame: the flicker check. Hold the camera still (a moving camera shows no colour). Orange: the pixel got brighter; blue: darker. How much colour is exact: full colour means the pixel changed by the percentage chosen here or more, half colour by half of it, and grey by nothing (so 0.25% shows the faintest flicker, 4% only big jumps). The rest of the picture is dim and grey; the probe balls keep their own colours. With a still camera the direct light never changes, so every coloured pixel is the GI changing as probes update. Try it with GI only, where nothing steady sits on top of the GI; it also works with the other debug views, and is off while the upscaler is on. The picture is stored at about 0.05-0.1% precision, so no pixel can change by less than that: on the 0.25% scale a single such step shows as a scattered faint speckle at 20-40% colour, and only pixels at full colour are really moving by 0.25% or more.",
    ddgiRestart:
        'Throws away what the probes have learned and warms them up again from black, the way a scene starts. Useful to watch how fast the light settles.',
    ddgiSaveBake:
        "Saves what the probes have learned to a file (.ddgibake) your browser downloads. Load it later, in this scene with the same settings, to start with finished GI instead of warming up. The file also records the scene it was made for, so a changed scene is noticed when it loads.",
    ddgiLoadBake:
        "Loads probes saved with the button above (or a .ddgibake file dropped onto the page), in place of the ones running now; they keep updating from there. A file made with other DDGI settings (another probe grid, light groups, the camera probes on or off) is refused, and the message says what differs. A file made before the scene changed (a light, an object, a material) still loads: the probes start from it and update to the new scene. Add warmup=0 to the link to start without the warm-up, so you can see the file's light arrive.",
    lpgBounces:
        'How many times light bounces before it is stored in the probes. 0 is direct light only, 1 adds one bounce (the usual choice), 2 adds a second. More bounces take longer to bake.',
    lpgProbes: 'The most probes the grid may use. More probes capture finer lighting changes but bake slower. The spacing is widened to fit.',
    lpgBake: 'Throw the baked lighting away and bake again, with the current settings.',
    vxgiBounces:
        'How many bounces of light are cached in the voxel copy of the scene. 1 is the usual choice. VXGI only bounces the sun and lamps: it takes no light from the sky.',
    wipeToggle:
        'Split the screen between two techniques on the same view. The left side shows one, the right side the other, with a white line between them. Drag the line to compare.',
    wipeLeft:
        'What is drawn left of the white line. "No GI" is direct light only. "Cycles reference" is a path-traced picture of a saved camera view, the right answer to compare against.',
    wipeRight: 'What is drawn right of the white line.',
    compareReference:
        'Split the screen: DDGI on the left, the path-traced Cycles picture of this camera view on the right, at the same exposure and tone mapping. The camera moves to the exact spot, field of view and picture shape the reference was rendered with (dark bars fill the rest of the window), so the two sides line up; move the camera and they no longer do. Its materials are plain matte, so shiny spots differ a little.',
    upscaleToggle:
        'Draws the picture at a lower resolution and rebuilds it to full size with @pmndrs/upscaler, which saves GPU time. Works with No GI and the light probe grid; VXGI has its own anti-aliasing, so this is switched off while VXGI is on screen.',
    upscaleQuality: 'How much smaller the picture is drawn. Quality is a little smaller and looks nearly sharp; Ultra performance is much smaller and softer.',
    statsToggle:
        "Shows the frame time, the GPU time of one frame, and what DDGI is doing (probes updated, its GPU time, memory, the adapter's CPU time). 'What the stats mean', just below, explains each line. GPU times need a GPU that supports timestamps.",
    statsExplain: 'What each line of the stats box (top left of the picture) means.',
    panelHide: 'Hides this panel so the picture fills the window. Press H, or the button at the top right, to bring it back.',
    wipeLine: 'Drag this line left or right to compare the two techniques.',

    // The stats box, top left (each line explains itself on hover).
    statsFrame: 'Time from one drawn frame to the next, and frames per second. 16.7 ms is 60 frames per second.',
    statsGpu: "GPU time of three.js's own passes in one frame (drawing the scene, shadows, the final picture), averaged over the last few frames. DDGI's probe updates are listed separately below.",
    statsProbes:
        'How many DDGI probes were updated in the last frame, out of all of them, and how long it takes at that rate until every probe has had its turn.',
    statsBudget:
        "Why DDGI updates this many probes a frame. When it starts it measures, with the GPU working at full speed, how many probes fit in the update budget, then updates that many every frame; a new budget is worked out from those measurements. It doesn't chase the measured time frame by frame: a GPU with little to do slows down to save power, which makes the same work look slower without costing the frame anything. It updates fewer only when frames arrive late ('slowed'), for example when the GPU gets hot and slows down, and goes back up a while after frames are on time again. It never goes below the number that updates every probe within 4 seconds, except on a slow GPU, where it stops at twice what the budget buys. 'measured cost' is what it measured: a fixed part per frame plus a part per probe. If the scene's shape changes it says so; Start the probes over to measure again.",
    statsDdgiUpdate:
        "GPU time of all of DDGI's passes in a frame: tracing rays from the probes (trace), blending the results into the probes (blend), writing them out for drawing (write), and, while GPU timing is on, a small pass that counts probes per state for these stats (census). The budget you chose is shown beside it. The budget was measured with the GPU at full speed; when the GPU has little to do it slows down, so this can read above the budget while frames are on time. That is expected and costs the frame nothing.",
    statsRefit:
        "GPU time to bring the ray-tracing tree up to date with where every object is, every frame. Cheap for a still scene; it matters once things move.",
    statsMemory:
        "GPU memory the GI library owns: its copy of the scene (positions, normals, materials), the ray-tracing trees, and DDGI's probes. three.js's own memory is not counted.",
    statsSync:
        "CPU time the three.js adapter spends per frame looking for changes in the scene and sending them to the GI library (target: 0.2 ms for a still scene), and then recording the GI work.",
    statsStates: 'How many probes are in each state, counted on the GPU a few frames ago.',
    statsRecompiles:
        'GPU programs the GI library had to compile while you used the viewer. It should stay at 0: moving, dimming, recolouring, switching lights, changing the time of day, moving, adding and swaying objects and flying the camera only change data. Starting DDGI over with light groups, the sun layer or the probes that follow the camera builds a new DDGI, which counts from 0 again. One exception: the "GI albedo at hits" debug view builds its own GPU program the first time it is shown, through the GI library, so the count reads 1 from then on; that one is expected.',

    // Lights, motions, time of day, light groups and the response meter (T2.8).
    lightSelect:
        "Choose one of the scene's lights to change. You can also click a light's marker in the picture (a small ball, or a cone for a spotlight). The chosen light gets arrows you can drag to move it.",
    lightOn: 'Switches the light on or off. Off sets its brightness to 0 rather than removing it, so nothing has to be rebuilt; GI follows within a frame or two.',
    lightIntensity:
        "How bright the light is, on a stepped scale: each notch to the right is about twice as bright. The number is in the light's own unit (candela for lamps, lux for the sun). GI follows: the probes near the light are updated first.",
    lightColor: 'The colour of the light. GI picks up the new colour in the bounced light within a few frames.',
    lightRange:
        'How far the light reaches, in metres: its light fades out to nothing at this distance. 0 means no limit (light falls off naturally with distance). GI uses the same reach.',
    lightCone: "How wide the spotlight's cone is, in degrees from its centre to its edge.",
    lightPenumbra: "How soft the edge of the spotlight's cone is: 0 is a hard edge, 1 fades all the way from the centre.",
    lightShadows:
        "Whether the light casts shadows. GI follows at once. three.js has to rebuild its own drawing programs when a light's shadow is switched, so the picture may pause for a moment the first time.",
    lightGroup:
        'Which light group the light belongs to in GI (0 to 3). With light groups on (below), each group can be dimmed or tinted in GI at once, with no update needed.',
    lightGizmoMove: 'Drag the arrows on the chosen light to move it. Drag a square to move it within a plane.',
    lightGizmoTurn: 'Drag the rings on the chosen spotlight to turn it and aim it somewhere else.',
    lightPutBack: 'Puts the chosen light back exactly as the scene has it: place, aim, brightness, colour, reach and group.',
    motionSelect:
        "Scripted movements from the scene file. Lamps: light sweep, a lamp gliding at walking pace (1 metre a second) and back; light step, the same lamp jumping between two places every 3 seconds. Objects: a door, a cabinet or a cabinet door, a tree in the wind. In Bistro: the camera's flight down the street. The ones that jump instead of gliding are for seeing how fast GI catches up (the response meter below uses them). Choose one, then press Play.",
    motionPlay: 'Plays the chosen movement, over and over. Stop puts what moved back where it was.',
    timeOfDayToggle:
        "Moves the sun with the hour and the day of the year below; the sky and GI follow. Turning it on also turns on DDGI's sun layer, which makes changes of the sun's brightness and colour show in GI at once; DDGI starts over for that, when you turn it on and again when you turn it off (a few seconds; Sponza about 10). Moving a slider turns it on. The time it opens at puts the sun exactly where the scene has it.",
    timeOfDayHour:
        'The time of day (sun time: the sun is highest at 12:00). The sun moves along its real path for the chosen day. Near sunrise and sunset the sky model stops at 5 degrees above the horizon, so the viewer fades the sun and sky out towards night instead.',
    timeOfDayDay: 'The day of the year, which sets how high the sun climbs: highest around 21 June, lowest around 21 December.',
    timeOfDayPlay: 'Plays one day from just before sunrise to just after sunset in 10 seconds, to watch GI follow a moving sun.',
    lightGroupsCount:
        "Gives GI a separate layer for each light group (0 to 3, set per light above) and one for the sky and glowing surfaces, so each can be dimmed or tinted at once with the sliders below. Off is the default: each layer costs memory and a little GPU time per probe. Changing this starts DDGI over (a few seconds).",
    lightGroupWeight:
        "Dims or tints this group's light in GI only. Instant: no update needed, nothing is traced again. The lamps' own direct light in the picture doesn't change; switch to GI only to see just the bounced light.",
    environmentWeight:
        "Dims or tints the light GI gets from the sky and from glowing surfaces. Instant: no update needed, nothing is traced again. The sky you see doesn't change.",
    meterPickRegion:
        'Drag a rectangle over the picture to choose what the meter watches: it averages that part of the picture each frame. Without one it watches the whole picture.',
    meterWholePicture: 'Watch the whole picture instead of a picked rectangle.',
    meterMeasure:
        "Measures how fast GI follows a step: one jump of a motion's lamp or object (choose which above; in two-room, jump 1 takes the lamp out of the lit room and jump 2 brings it back; for the door, jump 1 shuts it), and the meter plots the picked region frame by frame, in GI only. It first raises DDGI's budget to 16 ms for 4 seconds so the light has settled (a test mode, not what a game would do), waits 4 seconds at your budget, makes the jump, records 2 seconds, then raises the budget again until the picture stops changing (at least 4 seconds, at most 30) to find the settled picture. Keep the camera still: moving it stops the measurement. About 15 to 40 seconds in all.",
    meterCancel: 'Stops the measurement and gives DDGI its normal budget back.',
    meterPlot:
        "The picked region's brightness frame by frame after the step (frame 1 is the jump's own frame). Grey dashed: before the step; white: settled; the shaded band is within 5% of settled. Orange line: the first frame that changed by more than 1%; green line: the first frame within 5% of settled.",
    meterSaveCsv:
        "Saves the measurement as a CSV file: one row per frame (frame, milliseconds since the step, the region's brightness, its change from before and its distance from settled, probes updated), with the marks and how it was measured at the top.",

    // Moving things: the M3 scenarios (T3.8).
    scenarioSelect:
        'Pick a scenario to watch GI follow something that moves: a cabinet sliding along a wall, a door shutting between two rooms, a cabinet door opening (and cabinets you add), a tree in the wind, or a flight down a street. It opens the scene at the right camera view and starts the movement; the text below says what to watch for. The stats box (top left) shows the recompile counter, which should stay 0.',
    scenarioPlay: "Plays the scenario's movement again from the start, or stops it (whatever moved goes back where it was).",
    scenarioMeasure:
        "Measures how fast GI follows this scenario's step version (the door jumping shut, the cabinet jumping 1 m, the tree jumping to a full lean) with the response meter below: it switches to GI only, settles the light, makes the jump, and plots the picture's brightness frame by frame. Keep the camera still. About 15 to 40 seconds. When the step turns the picture black (the door shutting, seen from the dark room), \"within 5% of settled\" never comes, since 5% of nothing is nothing: read the curve, which should drop to the floor within the first second.",
    windStrength:
        "How strong the wind is: at 100% the strongest gusts move the top of the tree about 1.5 m, at 0% it stands still. Only the tree's shape changes: GI refits its ray-tracing tree each frame on the GPU, nothing is rebuilt and nothing recompiles.",
    cabinetAdd:
        'Adds a small cabinet on the floor to the right of the run, as a new shape: GI copies it and builds it a ray-tracing tree of its own on the GPU, so it casts shadows and bounces light within a frame or two.',
    cabinetCopy:
        'Adds one more cabinet of the same shape as the last one added: GI only places that shape again, which is cheaper than a new one. Up to 100 cabinets in all.',
    cabinetClear: 'Takes every added cabinet away again.',
    ddgiFollowCamera:
        "Adds a second, finer set of DDGI probes, 1.5 m apart (32 by 12 by 32 of them, about 46 m across), that moves with the camera in front of the scene's own probes, which fill in beyond it. Meant for big outdoor scenes like Bistro, whose own probes are 4 m apart. Probes that come in ahead of the camera fade in over a few seconds. Turning it on or off starts DDGI over (a few seconds).",
} as const;

export type HelpKey = keyof typeof HELP;

export const DEFAULT_HELP = 'Hover over any control to see what it does.';
