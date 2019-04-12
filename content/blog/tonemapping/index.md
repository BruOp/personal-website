---
title: Tone Mapping
date: "2019-04-12T22:12:03.284Z"
description: A guide to adding tone mapping to your physically based renderer
---

## Introduction

In [the last post](/blog/exposure), I explained how to can calculate the exposure of the HDR image produced by a physically based renderer like the one I've been slowly building using [BGFX](). This exposure allows us to basically "calibrate" our image such that we shift the luminance falues to a range where we can now apply a tone curve. In case you didn't read that, here's the diagram from the last post:

[DIAGRAM]

A tone curve is a function that takes a luminance value (or an individual color channel) as input and spits out a value between [0, 1], which is what our display expects. There are [many different curves]() that fulfill this function, but most have the following features:

- A "shoulder" which is meant to map larger luminance values to values approaching 1, but this convergence is typically asymptotic.
- A "foot", which is mean to control how the lower ranges of luminance are displayed. The foot is not as common or necessary as the shoulder, but it can help prevent darker portions of the image from being too dark.
- A linear or mostly linear portion, controls how your "mid tones" scale.

There are a few different curves that people have produced for these elements, and some are more customizable than others. Perhaps the most canonical are the [Reinhard curves].

## Reinhard Curves

There are basically two Reinhard curves that people have used over the years. The first, simpler one is as follows:

$$
L_d = \frac{L^\prime}{1+L^\prime}
$$

Where $L_d$ is the final display luminance, and $L^\prime$ is the exposure adjusted input luminance:

$$
L^\prime = \frac{L_{\text{in}}}{9.6 * L_{avg}}
$$

This is a really easy way to effectively prevent any luminance from exceeding the max value of 1.0, while scaling lower luminance values (where $L << 1$) linearly (e.g $L_d = L$). However, this is going to prevent you from ever actually having a luminance value of 1.0, so there is a second, modified equation that allows us a bit more control over when the luminance reaches 1.0:

$$
L_d = \frac{L^\prime * \left( 1 + \frac{L^\prime}{L_\text{white}^{2}} \right)}{1+L^\prime}
$$

Where $L_{white}$ is a user-controlled variable that allows us to set the saturation point. You can see how changing $L_{white}$ changes the function using [this desmos graph](https://www.desmos.com/calculator/h8lpdqtlxi).

For reference, here's three images that show gamma corrected outputs that use three different curves: linear (e.g. no curve applied), Reinhard and adjusted Reinhard:

[THREE IMAGES]

While these curves were some of the most popular to use, they have their limitations. John Hable's blog post from 2010 does a good job of demonstrating and discussing why people have started moving away from Reinhard curves. You can see in our reference images that the simple Reinhard really removes most of our highlights, while also washing out our blacks. The adjusted reinhard has better highlights, but that's completely determined by our choice of $L_{white}$. If our value is too low, then we'll get more and more blown out features like with the linear version, but as $L_{white}$ grows larger, we'll end up with something very similar to the simple Reinhard.

[IMAGES of different values of Lwhite]

## Filmic Curves

While Reinhard is clearly better than nothing, the industry has appeared to move towards other curves that are more customizable, especially in terms of how they behave towards the extremes. Romain Guy created an excellent [shader toy](https://www.shadertoy.com/view/WdjSW3) demonstration of these different curves, which I've modified slightly, adding the two reinhard curves.

To apply the curves, we can write a simple fragment shader that takes our HDR PBR output, the average luminosity (store in a 1x1 texture) and writes the output to the backbufffer:

```glsl
$input v_texcoord0

#include "common.sh"

SAMPLER2D(s_texColor, 0);
SAMPLER2D(s_texAvgLum, 1);

void main()
{
  vec3 rgb = texture2D(s_texColor, v_texcoord0).rgb;
  float lum = texture2D(s_texAvgLum, v_texcoord0).r;

  vec3 Yxy = convertRGB2Yxy(rgb);

  float middleGray = u_tonemap.x;
  float whiteSqr   = u_tonemap.y;

  float lp = Yxy.x * middleGray / (lum + 0.0001);
  Yxy.x = reinhard2(lp, whiteSqr);

  rgb = convertYxy2RGB(Yxy);

  gl_FragColor = toGamma(vec4(rgb, 1.0) );
}

```

The code is fairly simple, and the process is simple:

1. Transform the RGB value into CIE xyY color space, where the Y is the luminosity
2. Adjust for exposure
3. Scale Y (and only Y) using the tone curve
4. Perform the reverse transformation (xyY -> RGB)
5. Apply gamma correction and write the results to the backbuffer

Using the different curves, we'll get different results:

[SERIES OF IMAGES]

Each of the tone curves will produce a slightly different "look" to the final image, but it should be obvious that the filmic curves produce a vastly different image from the others. The filmic curves produce a less "washed out" image than the Reinhard curves.

I should also note that the choice to apply the curve to the Luminance values exclusively rather than each RGB channel individually is a choice. John Hable [advocates the latter approach](http://filmicworlds.com/blog/filmic-tonemapping-with-piecewise-power-curves/), but this will result in shifts in Hue. [This presentation by the Frostbite team]() provides an example of hue preserving vs non-hue preserving tonemapping. In the end, it's a choice you'll just have to make! To me it makes more sense to continue working with luminance values, but there are arguments why that's suboptimal and won't produce the final image you want.

## A note on Gamma Correction

I won't go into gamma correction in great detail, but it is an _essential part_ of not only tone mapping, but working with color values in computer graphics in general. But in order for our image to appear correctly even after tone mapping, we need to apply a gamma correction. If you don't know much about Gamma or what I'm talking about, here's a few great reads:

[LIST of entries]

## Wrapping up

If you read the last post, recall that I presented the following image as output from my toy physically based renderer, **without** tonemapping:

[IMAGE]

Note that this image has Gamma correction, but the luminance values are still outside the display's range. It is technically physically correct, but our display doesn't care. Here is the same scene with tonemapping applied using :

[GOOD SDR IMAGE]

Hopefully the stark difference demonstrates the need for tonemapping if you are attempting to implement PBR shading _with actual physical values_ for your lights.