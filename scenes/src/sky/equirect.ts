/**
 * The sky's equirectangular layout lives in the core since T1.4, so the
 * platform's own sky code and the sky files share one definition. This file
 * keeps the old import path working for the tools that name it.
 */

export * from '../../../packages/ddgi-core/src/platform/sky/equirect.ts';
