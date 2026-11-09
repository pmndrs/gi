/**
 * Which screen channels the attached techniques read (RC1.1): the union of
 * their `requirements.screen`, for `GIPlatform.screenNeeds`, and the check
 * `encodeScreen` makes before handing a screen to them.
 */

import type { FrameScreen, OptionalScreenChannel } from '../contract/frame.ts';
import type { ScreenNeeds } from '../contract/platform.ts';
import type { TechniqueRequirements } from '../contract/technique.ts';

/** The optional channels, in the order messages list them. */
export const OPTIONAL_SCREEN_CHANNELS: readonly OptionalScreenChannel[] = ['velocity', 'albedo', 'instance'];

export const NO_SCREEN_NEEDS: ScreenNeeds = { screen: false, velocity: false, albedo: false, instance: false };

/** One technique's screen requirement as a `ScreenNeeds`. */
export function screenNeedsOf(requirements: TechniqueRequirements): ScreenNeeds {
    const screen = requirements.screen;
    if (screen === undefined || screen === false) return NO_SCREEN_NEEDS;
    if (screen === true) return { screen: true, velocity: false, albedo: false, instance: false };
    return { screen: true, velocity: screen.velocity === true, albedo: screen.albedo === true, instance: screen.instance === true };
}

/** The union of several techniques' screen requirements. */
export function mergeScreenNeeds(all: readonly TechniqueRequirements[]): ScreenNeeds {
    let merged = NO_SCREEN_NEEDS;
    for (const requirements of all) {
        const needs = screenNeedsOf(requirements);
        merged = {
            screen: merged.screen || needs.screen,
            velocity: merged.velocity || needs.velocity,
            albedo: merged.albedo || needs.albedo,
            instance: merged.instance || needs.instance,
        };
    }
    return merged;
}

/** The channels `needs` asks for that `screen` doesn't have, in a fixed order. */
export function missingScreenChannels(needs: ScreenNeeds, screen: FrameScreen): OptionalScreenChannel[] {
    return OPTIONAL_SCREEN_CHANNELS.filter((channel) => needs[channel] && screen[channel] === undefined);
}
