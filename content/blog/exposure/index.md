---
title: Tone Mapping using a Luminance Histogram
date: "2019-03-28T22:12:03.284Z"
description: A guide to adding tone mapping to your physically based renderer
---

## Introduction

Recently, I've been building a small renderer using the cross-platform rendering library [bgfx](https://github.com/bkaradzic/bgfx) as a learning exercise. To start, I decided to load in some [glTF]() models and write a basic PBR shader using the Cook-Torrance model described in the [glTF specification](). The implementation is incomplete and it renders the material using only punctual lights. It's a single vertex + fragment shader. I loaded up the [FlightHelmet]() model, added two lights on each side, started up my program and saw this:

[IMAGE OF SHIT LDR IMAGE]

You can see that the lit areas are basically just the color of the light sources, with ugly, uniform splotches removing all of the detail we'd expect to get from the normal maps and albedo textures. This is not unexpected. It may not be obvious why this isn't a bug and why we'd expect our fragment shader to produce such an ugly image, so let me quickly explain.

### Physically Based Lighting

In physically based rendering, objects are rendered using _physically based units_ from the fields of [Radiometry]() and [Photometry](). The [rendering equation]() is meant to produce the [radiance]() for each pixel. This means, for instance, that the lights are using the photometric unit of [lumens](), and both are set to emit 800 lm, but it could be much higher. The sun, a directional light, illuminates the earth's surface with ~120,000 [lux](). This means that when we solve the rendering equation for these objects, we're going to end up with values that are effectively unbounded and we may end up with radiance values that differ by several orders of magnitude in the same frame. Additionally, all of our calculations are taking place in linear space -- that is to say, that an RGB value corresponding to (1.0, 1.0, 1.0) corresponds to half as much radiance as a value of (2.0, 2.0, 2.0).

This is a problem however, because (most) of our displays work differently. Displays expects our frame buffer to contain RGB values in the [sRGB color space]() that are between 0 and 1, with (1.0, 1.0, 1.0) corresponding to white.<sup>[1](#note_1)</sup> So any RGB that our fragment shader produces are clamped to [0, 1] when they are written to the 32-bit back buffer. That's why everything appears to be basically white!

## Tone Mapping

The solution then, is to take our physical, unbounded HDR values and map them first to a LDR linear space [0, 1], and then finally apply [gamma correction]() to produce the sRGB value our displays expect.

The way we perform the first step, mapping the HDR values produced by our fragment shader to linear LDR values is usually called _Tone Mapping_. There are a few different ways to perform it, and the way you choose to do it will have an effect on the "look" of your final frame, but the basic steps are:

1. Render your scene into a framebuffer that supports HDR values i.e. a floating point buffer with 16-bits or 32-bits per channel. Which one you choose is (as always) a tradeoff between precision and memory, but for my scene I'm going to go with a RGBA32F framebuffer. Make sure your output is in linear space.
2. In a separate render pass, produce an LDR buffer by:
   1. Scale the input using the exposure of the image to obtain a "calibrated" RGB value
   2. Scale the calibrated value using a Tone Curve, inputting either the RGB or the Luminance value of the fragment
   3. Apply gamma correction to the scaled value and write this result to the back buffer.

If you're anything like me, you are probably asking yourself:

- How do I actually obtain the "calibrated" fragment value? What does calibration even mean in this context?
- What is a Tone Curve and which one should I use?
- What is the difference between using the luminance value as input to the tone curve vs the RGB value?

I'm going to tackle these one at a time, focusing first on calibration. Once we've made our choices, we'll revisit this task list and go into implementation.

### Exposure

When a human eye views a scene, it will naturally dilate to adjust to the amount of light reaching it. Similarly, photographers have several ways to control the amount of light reaching the sensors, such as the aperture size (f-stop) and shutter speed. In photography, these controls correspond to the [exposure value](), or $EV$, which is a logarithmic representation of the luminance. Increasing the $EV$ by a value of +1 results in a doubling of the luminance. I won't spend too much time on $EV$ as I can't s

In our context, the exposure linearly scales our the scene luminance to simulate how much light is actually hitting the sensor. It's up to us to actually provide the exposure value, but in order to avoid having it set by the user/artist, we're going to use the average scene luminance to derived from our HDR buffer to calculate the exposure.

There are actually quite a few different ways to calculate exposure from average scene luminance, many of which are explained in this excellent post by [Krzysztof Narkowicz](https://knarkowicz.wordpress.com/2016/01/09/automatic-exposure/). Turns out that photography and film has been dealing with this problem for a much longer time that real time rendering, so there is an entire literature associated with this specific problem.

For this post, I'm going to use the method described in [Lagard and de Rousiers, 2014](https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/course-notes-moving-frostbite-to-pbr-v2.pdf) (pg. 85). Long story short, first we can calculate the luminance value that will saturate the sensor, $L_{max}$, then use that to scale our scene luminance:

$$
EV_{100} = \log_{2} \left( \frac{S \times L_{avg}}{K} \right)
$$

$$
L_{max} = \frac{78}{qS} \times 2^{EV_{100}} \\
$$

$$
\text{H} = \frac{1}{L_{max}} \\
$$

Where $S$ is the Sensor sensitivity, $K$ is the reflected-light meter calibration constant, $q$ is the lens and vignetting attentuation, $H$ is the exposure and $L_{avg}$ is the average scene luminance.

If we were fully modelling a physical camera, we might need to use different $S$ values to offset the loss of light when change aperture, which has effect on the depth of field. But since we aren't worrying about that, we'll use $S=100$. Meanwhile, it seems that Canon, Nikon and Sekonic all use $K = 12.5$ and it seems most rendering engines follow suit. Finally, $q=0.65$ appears similarly ubiquitous. If you want to know a bit more about what these quantities actually represent, the previously referenced [Lagard and de Rousiers, 2014](https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/course-notes-moving-frostbite-to-pbr-v2.pdf) has more detail, as does the [Filament](https://google.github.io/filament/Filament.html#physicallybasedcamera) documentation.

We can now rewrite our previous equation for $L_{max}$ more simply:

$$
L_{max} = 9.6 \times L_{avg}
$$

Our final scaled color is then simply:

$$
c_{rgb}^\prime = \text{Exposure} \times c_{rgb} = \frac{c_{rgb}}{L_{max}}
$$

Note that this value is still not clamped to [0, 1], and so we will still potentially get a lot of clipping.

Additionally, we haven't discussed how to actually calculated the average luminance. There are two primary ways to do so:

- Use a geometric average, obtained by repeatedly downsampling the luminance of our HDR image similar to how we would create a mip map chain.
- Create a histogram of some static luminance range.

The geometric average is susceptible to extreme values being over-represented in the final luminance value, so instead we're going to construct the histogram. This allows us more control (if we desire it) over how extreme values influence our average.

### Tone Curves

Once we have scaled our scene luminance by the exposure, we can now apply a _tone curve_. A tone curve is literally just a function that takes our exposed color value (that we obtained above) to

## Notes

1. This is not true for HDR displays. Those displays use other color spaces that newer games are starting to support. [LINK]

<!-- I've read about many different techniques in computer graphics on blogs, twitter and my copy of [Real Time Rendering](), but I've rarely actually gone through and implemented any of them. My previous attempts to start had be -->
