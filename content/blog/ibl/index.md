---
title: Image Based Lighting with Multiple Scattering
date: "2019-08-19T15:00:00Z"
description: A guide to adding image based lighting that uses a multi-scattering
---

- [Introduction and Background](#introduction-and-background)
- [Image Based Lighting Challenges](#image-based-lighting-challenges)
- [Lambertian Diffuse Component](#lambertian-diffuse-component)
- [Importance Sampling](#importance-sampling)
- [Split Sum Approximation](#split-sum-approximation)
  - [Pre-filtered Environment Map](#pre-filtered-environment-map)
  - [Environment BRDF](#environment-brdf)
- [Single Scattering Results](#single-scattering-results)
- [Accounting for Multiple-Scattering](#accounting-for-multiple-scattering)
  - [Metals](#metals)
  - [Dielectrics](#dielectrics)
    - [Roughness Dependent Fresnel](#roughness-dependent-fresnel)
    - [GLSL Shader Code](#glsl-shader-code)
- [Future Work](#future-work)
- [Source Code](#source-code)

## Introduction and Background

Recently I decided to implement image based lighting in BGFX, since I had never implemented image based lighting before and it's a great way to get assets authored for PBR looking really great. As I started reading I realized that there was a lot of work done on this in the past few years built upon bit by bit, and that it might be useful to others to have a reference for implementation from start to finish. I've document the steps involved in implementation, starting with some background, then [Karis's 2014 paper](https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf) explaining Unreal's IBL implementation, and then new approaches from [Fdez-Agüera](http://www.jcgt.org/published/0008/01/03/paper.pdf) and others to address the error present in existing models.

Let's start by defining the equation which provides the outgoing light from a point $\mathbf{p}$ in the viewing direction $\mathbf{v}$:

$$
L_o(\mathbf{v}) = \int_{\Omega} f_{\text{brdf}}(\mathbf{v}, \mathbf{l}) L_i(\mathbf{l}) \langle\mathbf{n} \cdot \mathbf{l}\rangle d\mathbf{l}
$$

Where $\mathbf{v}$ is our viewing angle, $f_{\text{brdf}}(\mathbf{v}, \mathbf{l})$ is our bidirectional reflectance distribution function (BRDF), $L_i(\mathbf{l})$ is the radiance incident on point $\mathbf{p}$ from direction $-\mathbf{l}$, and $\mathbf{n}$ is the normal vector at point $\mathbf{p}$. The hemisphere $\Omega$ is aligned with $\mathbf{n}$, and $\mathbf{h}$ is the is the halfway vector, halfway between $\mathbf{l}$ and $\mathbf{v}$.

I've omitted $\mathbf{n}$ and $\mathbf{h}$ as function arguments, but in reality they are also required to evaluate the BRDF. Here's a diagram of our different vectors:

![A diagram showing the different vectors we're using](./hemisphere_diagram.png)

For our BRDF, we will take the popular approach of using a Lambertian diffuse lobe and Cook-Torrance microfacet model for our specular lobe:

$$
\begin{aligned}
f_{\text{brdf}}(\mathbf{v}, \mathbf{l}) &= (f_{\text{specular}}(\mathbf{v}, \mathbf{l}) + f_{\text{diffuse}},
\\\\
f_{\text{specular}}(\mathbf{v}, \mathbf{l}) &=
    \frac{
        D(\mathbf{h}) F(\mathbf{v}\cdot\mathbf{h}) G(\mathbf{v},\mathbf{l},\mathbf{h})
    }{
        4 \langle\mathbf{n}\cdot\mathbf{l}\rangle \langle\mathbf{n}\cdot\mathbf{v}\rangle
    },
\\
f_{\text{diffuse}} &= \frac{c_{\text{diff}}}{\pi},
\end{aligned}
$$

Where $D$ is our normal distribution function (NDF), which tells us what proportion of our perfectly reflecting microfacets will have normals aligned with $\mathbf{h}$, therefore reflecting the light from $-\mathbf{l}$ in the direction $\mathbf{v}$. $F$ is the Fresnel term which defines what proportion of light is reflected as opposed to refracted into the surface. $G$ is the "self shadowing term", which defines what proportion of our microfacets will be occluded by the surrounding surface along the direction $\mathbf{l}$.

For more detail on this model, there are many resources but [Naty Hoffman's talk](https://blog.selfshadow.com/publications/s2013-shading-course/hoffman/s2013_pbs_physics_math_slides.pdf) from this [2013 Siggraph workshop](https://blog.selfshadow.com/publications/s2013-shading-course/) is very helpful in explaining this BRDF if you've never seen it before. I'll also go ahead and define our functions explicitly, to dispell any ambiguity as to which variation we're using here.

One extremely important detail is that the $f_\text{specular}$ lobe only accounts for a single scattering of energy. In reality, rougher surfaces will produce multiple scattering events. This is especially important metals, where reflection dominates. We will see the consequences of only including a single scattering event in our BRDF model later.

For our NDF $D$, we're using the [GGX lobe](https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf):

$$
D(\mathbf{h}) = \frac{
    \alpha^2
}{
    \pi ((\mathbf{n}\cdot\mathbf{h})^2(\alpha^2 - 1) + 1)^2
}
$$

where $\alpha = \text{roughness}^2$. Here, $\text{roughness}$, AKA perceptually linear roughness, will be an input into our shading model, stored as a channel in our textures.

Meanwhile, for our geometric shadowing/attenuation term $G$ we're using the height correlated Smith function for GGX:

$$
\begin{aligned}
V(\mathbf{v}, \mathbf{l}) &=
\frac{
    G(\mathbf{v},\mathbf{l},\mathbf{h})
}{
    4 \langle\mathbf{n}\cdot\mathbf{l}\rangle \langle\mathbf{n}\cdot\mathbf{v}\rangle
},
\\
V(\mathbf{v}, \mathbf{l}) &= \frac{0.5}{
    \langle \mathbf{n}\cdot\mathbf{l} \rangle \sqrt{(\mathbf{n}\cdot\mathbf{v})^2 (\alpha^2 - 1) + \alpha^2}
+
    \langle \mathbf{n}\cdot\mathbf{v} \rangle \sqrt{(\mathbf{n}\cdot\mathbf{l})^2 (\alpha^2 - 1) + \alpha^2}
}
\end{aligned}
$$

Notice we've expressed this in a way that incorporates the denominator of the BRDF into our $V$ term. For more detail on this term, I've found [this section](https://google.github.io/filament/Filament.html#materialsystem/specularbrdf) of the Filament documentation to be helpful.

Finally, our Fresnel term will use the Schlick approximation:

$$
\begin{aligned}
F(\mathbf{v}, \mathbf{l}) &= F_0 + (1 - F_0)(1 - \langle \mathbf{v} \cdot \mathbf{h} \rangle)^5
\\ &= F_0 (1 - (1 - \langle \mathbf{v} \cdot \mathbf{h} \rangle)^5) + (1 - \langle \mathbf{v} \cdot \mathbf{h} \rangle)^5
\end{aligned}
$$

Where $F_0$ is a material property representing the specular reflectance at incident angles ($\mathbf{l} = \mathbf{n}$). The second form will come in handy later. For dielectrics, this is taken to be a uniform 4% as per the [GLTF spec](https://github.com/KhronosGroup/glTF/blob/master/specification/2.0/README.md#materials), while for metals we can use the albedo channel in the texture:

```glsl
F_0 = lerp(vec3(DIELECTRIC_SPECULAR), albedo.rgb, metalness);
```

For the constant diffuse term $f_{\text{diffuse}}$, we'll be using the [GLTF spec](https://github.com/KhronosGroup/glTF/blob/master/specification/2.0/README.md#materials) for determining $c_{\text{diff}}$, which allows for a single albedo texture used by both metals and dielectrics:

```glsl
const float DIELECTRIC_SPECULAR = 0.04;
diffuseColor = albedo.rgb * (1 - DIELECTRIC_SPECULAR) * (1.0 - metalness)
```

## Image Based Lighting Challenges

Okay, so that's our BRDF and the reflectance equation we need to evaluate for every pixel covering our mesh. Let's turn our attention now to the hemisphere $\Omega$ we need to integrate across. In order to do so, we need to be able to evaluate $L_i(\mathbf{l})$ for any given $\mathbf{l}$. In real time applications, we have two approaches, often used together:

1. Describe $L_i(\mathbf{l})$ using a set of $N$ analytical lights. Then, we only need to evaluate the integral $N$ times, for each corresponding light direction $\mathbf{l}_i$.
2. Describe $L_i(\mathbf{l})$ using an _environment map_, often using a [cube map](https://en.wikipedia.org/wiki/Cube_mapping).

We'll be focused on item 2 in this post. While an environment map can provide much more realistic lighting, the naive evaluation of our hemispherical integral is not going to cut it for interactive computer graphics. To gain a quick understanding of the challenges involved, consider the following:

- With a typical microfacet model we have both specular and diffuse lobes.
- The Lambertian lobe is dependent on light incoming from every part of our hemisphere. We must sample as much of the hemisphere as possible.
- Sampling the hemisphere naively, we'll need potentially hundreds/thousands of texture reads.
- For the specular lobe, our roughness will determine what parts of the hemisphere are most important. Other parts will not be as important and will not help us converge on a solution.
- Additionally, each sample will require evaluation of the BRDF, which for the specular component, is not "cheap".

Doing all this work with every mesh is not going to cut it for anything but the simplest of scenes. Additionally, sometimes we'll be redoing work -- for instance, calculating the diffuse irradiance for each direction really only needs to happen once for each environment map!

So instead of doing all that in every frame, we're going to try to pre-compute as many parts of the integral as possible. In order to do so, we'll need to identify what we can precalculate and have to make some approximations!

## Lambertian Diffuse Component

First, since it's easier, let's look at the diffuse part of our reflectance integral:

$$
\begin{aligned}

L_{\text{diffuse}}(\mathbf{v})
&= \int_{\Omega} \frac{c_{\text{diff}}}{\pi} L_i(\mathbf{l}) \langle\mathbf{n} \cdot \mathbf{l}\rangle d\mathbf{l}
\\&= c_{\text{diff}} \int_{\Omega} \frac{1}{\pi} L_i(\mathbf{l}) \langle\mathbf{n} \cdot \mathbf{l}\rangle d\mathbf{l}
\end{aligned}
$$

Since our integral is not dependent on any material properties or view direction, and instead dependent only on $\mathbf{n}$, we can solve this integral for every direction $\mathbf{n}$ and store the results in a cube map, allowing us to look up the irradiance for any normal direction later. Note that since we're not using an NDF or anything like that, we can get away with simply uniformly sampling our hemisphere. Then we just multiply it by $c_{\text{diff}}$ later on and we're done!

In BGFX, I implemented this using compute shaders, using thread groups with a third dimension corresponding to a face of the cube. I'm not certain this is actually a good idea as it result in poor coherency, but I didn't have time to benchmark against just dispatching a different job for each face. This is left as an exercise for the reader.

Here's the shader code used to generate a 64x64 irradiance map inside of a compute shader:

```glsl
#define TWO_PI 6.2831853071795864769252867665590
#define HALF_PI 1.5707963267948966192313216916398

#define THREADS 8

SAMPLERCUBE(s_source, 0);
IMAGE2D_ARRAY_WR(s_target, rgba16f, 1);

NUM_THREADS(THREADS, THREADS, 6)
void main()
{
    const float imgSize = 64.0;
    ivec3 globalId = gl_GlobalInvocationID.xyz;

    vec3 N = normalize(toWorldCoords(globalId, imgSize));

    vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    const vec3 right = normalize(cross(up, N));
    up = cross(N, right);

    vec3 color = vec3_splat(0.0);
    uint sampleCount = 0u;
    float deltaPhi = TWO_PI / 360.0;
    float deltaTheta = HALF_PI / 90.0;
    for (float phi = 0.0; phi < TWO_PI; phi += deltaPhi) {
        for (float theta = 0.0; theta < HALF_PI; theta += deltaTheta) {
            // Spherical to World Space in two steps...
            vec3 tempVec = cos(phi) * right + sin(phi) * up;
            vec3 sampleVector = cos(theta) * N + sin(theta) * tempVec;
            color += textureCubeLod(s_source, sampleVector, 0).rgb * cos(theta) * sin(theta);
            sampleCount++;
        }
    }
    imageStore(s_target, globalId, vec4(PI * color / float(sampleCount), 1.0));
}
```

Note that you may not need a such a small step size. In fact, depending on the size of your environment map, you may be wasting your time a bit. I was struggling with ringing artifacts with certain environment maps, so I opted to use 1 sampler/degree which seemed to help. Since the operation is done once per environment map, offline, I didn't want to spend too much time tweaking performance.

## Importance Sampling

In order to integrate our radiance equation, we'll need to numerically integrate across our hemisphere. Using the Monte Carlo with importance sampling will provide the following equation:

$$
\int_{\Omega} f_{\text{brdf}}(\mathbf{v}, \mathbf{l}) L_i(\mathbf{l}) \langle\mathbf{n} \cdot \mathbf{l}\rangle d\mathbf{l}
\approx
\frac{1}{N} \sum_{k=1}^{N} \frac{f_{\text{brdf}}(\mathbf{v}, \mathbf{l}_k) L_i(\mathbf{l}_k) \langle\mathbf{n} \cdot \mathbf{l}_k\rangle}{\text{pdf}(\mathbf{v}, \mathbf{l}_k)}
$$

Where $\text{pdf}(\mathbf{v}, \mathbf{l}_k)$ is the probability distribution function we use for sampling our hemisphere. Intuitively, the idea behind having the PDF in the denominator is that if a direction is more likely to be sampled than others, it should contribute less to the sum than a direction that is incredibly unlikely to be chosen.

We could naively sample our hemisphere randomly, but at low roughness our specular component will converge to the actual solution very slowly. The same is true if we attempt to sample our hemisphere uniformly. To reason about this, consider a roughness close to zero, which corresponds to a perfect reflector. In this case, the only light direction that matters is the direction of perfect reflection $\mathbf{l}_{\text{reflection}}$ from $\mathbf{v}$ based on $\mathbf{n}$. _Every other direction will contribute nothing to our sum_. And if we do manage to sample $\mathbf{l}_{\text{reflection}}$, we still won't converge due to the $\frac{1}{N}$ term. At hight roughness, this is less of a factor as the distribution of our microfacet normals will increase, and we'll have to consider most parts of our hemisphere.

So instead, we'll choose a PDF that resembles our BRDF in order to decrease the variance. We'll actually choose our Normal Distribution Function $D$ as our PDF, the reasoning being that the NDF determines which directions contribute the most to the light leaving our larger surface patch. More correctly, we'll actually be sampling $D(\mathbf{h})\langle \mathbf{n} \cdot \mathbf{h} \rangle$, since [this is how the normal distribution is actually defined](http://www.reedbeta.com/blog/hows-the-ndf-really-defined/):

$$
\int_\Omega D(\mathbf{h}) \langle \mathbf{n} \cdot \mathbf{h} \rangle d\mathbf{l}s = 1
$$

If we want our PDF to be used for $\mathbf{l}$ instead of the half vector $\mathbf{h}$ then we'll need to include the Jacobian of the transformation from half vector to $\mathbf{l}$, $J(\mathbf{h})$:

$$
J(\mathbf{h}) = \frac{1}{4\langle \mathbf{v} \cdot \mathbf{h} \rangle}
$$

$$
\text{pdf}(\mathbf{v}, \mathbf{l}_k) = \frac{D(\mathbf{h}) \langle \mathbf{n} \cdot \mathbf{h} \rangle}{4\langle \mathbf{v} \cdot \mathbf{h} \rangle}
$$

You can also read section 5.3 of [Walter et al. 2007](http://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf), as well as [Legarde et al's course notes](https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf) for more detail.

It's also worth noting that we won't be sampling $\mathbf{l}$ directly and instead we'll sample our microfacet normal $\mathbf{h}$ and use it to find the relevant $\mathbf{l}$. To do so, we'll need to map two uniformly random variables in the interval $[0, 1)$, let's call them $\xi_1, \xi_2$, to $\phi$ and $\theta$ in spherical coordinates. Then, we can turn our $\phi, \theta$ into a cartesian direction $\mathbf{h}_i$ in world space. We can then use this to find $\mathbf{l}$ to sample our environment map to evaluate $L_i$.

The mapping from $\xi_1, \xi_2$ to $\phi$ and $\theta$ won't be derived here, but [Tobias Franke](https://www.tobias-franke.eu/log/2014/03/30/notes_on_importance_sampling.html) and [A Graphics Guy](https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-brdf/) have good blog posts that break this down step by step. Ultimately the mapping is as follows:

$$
\begin{aligned}
\phi &= 2 * \pi * \xi_1, \\
\theta &= \arccos \sqrt{\frac{1 - \xi_2}{\xi_2(\alpha^2 - 1) + 1}}
\end{aligned}
$$

Note that this assumes we're restricting ourselves to an isotropic version of the GGX, which is why the $\phi$ term is just randomly sampled. Note the $\alpha$ term in our equation for $\phi$, but it may not be obvious how this dependency causes the equation to behave. I created a [little Desmos demo](https://www.desmos.com/calculator/ueuitucusv) that you can play around with to get a sense of how this maps our $\xi_2$ to $\theta$. Notice that at low roughnesses, most of the $[0, 1)$ range will map to a small $\theta$, while at larger roughnesses we approach a standard cosine distribution, and we're much more likely to get directions "further" from $\mathbf{n}$.

Okay so, we now have a method of importance sampling for the entire range of roughness values. Here's some GLSL code that shows how we could sample a quasi-random direction $\mathbf{l_k}$:

```glsl
// Taken from https://github.com/SaschaWillems/Vulkan-glTF-PBR/blob/master/data/shaders/genbrdflut.frag
// Based on http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
vec2 hammersley(uint i, uint N)
{
    uint bits = (i << 16u) | (i >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    float rdi = float(bits) * 2.3283064365386963e-10;
    return vec2(float(i) /float(N), rdi);
}

// Based on Karis 2014
vec3 importanceSampleGGX(vec2 Xi, float roughness, vec3 N)
{
    float a = roughness * roughness;
    // Sample in spherical coordinates
    float phi = 2.0 * PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    // Construct tangent space vector
    vec3 H;
    H.x = sinTheta * cos(phi);
    H.y = sinTheta * sin(phi);
    H.z = cosTheta;

    // Tangent to world space
    vec3 upVector = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangentX = normalize(cross(upVector, N));
    vec3 tangentY = cross(N, tangentX);
    return tangentX * H.x + tangentY * H.y + N * H.z;
}
```

One thing in the above code I did not touch on at all is the Hammersley sequence, which is the method by which we create our random numbers $\xi_1, \xi_2$. It's a low-discrepancy sequence that I won't do any justice describing, so here's a [wikipedia link](https://en.wikipedia.org/wiki/Low-discrepancy_sequence#Hammersley_set).

We can further improve the rate of convergence by filtering our _samples_ as well. The idea is described in detail in [Křivánek and Colbert's GPU Gems article](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch20.html) and I will provide a quick summary. When using our importance sampling routine, we evaluate the PDF value for the sample, and if it is "small" then this corresponds to a direction we sample infrequently. To improve convergence, we can consider this equivalent to sampling a larger solid angle in our hemisphere. The inverse is also true: samples with higher PDF values will correspond to smaller solid angles. But how do we sample a larger or small solid angle in our environment without just having to integrate again? Well we can approximate this by sampling the mip chain of our environment map! It's a very clever trick, and drastically improves convergence. You'll see it demonstrated in some shader code later on.

Now that we can importance sample our hemisphere, we can now turn our attention back to the radiance equation and see which bits we can calculate and store to cut down on the amount of work we need to do each frame.

## Split Sum Approximation

In [Karis's presentation of the Unreal Engine PBR pipeline](https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf), he explains that we can actually approximate the specular part of our radiance sum by splitting it in two:

$$
\frac{1}{N} \sum_{k=1}^{N}
\frac{
    f_s(\mathbf{v}, \mathbf{l}_k) L_i(\mathbf{l}_k) \langle\mathbf{n} \cdot \mathbf{l}_k\rangle
}{
    \text{pdf}(\mathbf{v}, \mathbf{l}_k)
} \approx
    \left( \frac{1}{N} \sum_{k=1}^{N} L_i(\mathbf{l}_k) \right)
    \left( \frac{1}{N} \sum_{k=1}^{N} \frac{f_s(\mathbf{v}, \mathbf{l}_k) \langle\mathbf{n} \cdot \mathbf{l}_k\rangle}{\text{pdf}(\mathbf{v}, \mathbf{l}_k)} \right)
$$

As Karis notes, this approximation is exact for a constant $L_i$ term, and reasonable for "common environments". The way this approximation helps us is that we can store the two different sums separately!

### Pre-filtered Environment Map

The first sum is what's commonly referred to as the "Pre-filtered Environment Map" or just the "radiance map" in other papers. When we say "pre-filtering", what we're actually doing is evaluating and storing a value such that when we go to render our mesh, we can simply sample this stored value to obtain $L_i$. The idea is that we can skip having to integrate across our hemisphere in order to figure out the radiance reflected in $\mathbf{v}$, and instead just look it up in a prefiltered cube map.

Interestingly, Karis and others convolve this integral with the GGX by still using the GGX based importance sampling. This improves convergence, which is important if you're planning on having your environment map change over time and need to re-evaluate it outside the context of an offline pipeline. So really, we'll be performing the following integral instead of the one above:

$$
\frac{4}{\sum_{k=1}^{N} \langle \mathbf{n} \cdot \mathbf{l}_k \rangle }
\left( \sum_{k=1}^{N}
    \langle \mathbf{n} \cdot \mathbf{l}_k \rangle
    \frac{
        L_i(\mathbf{l}_k) \langle \mathbf{v} \cdot \mathbf{h} \rangle
    } {
        D(\mathbf{h}) \langle \mathbf{n} \cdot \mathbf{h} \rangle
    }
\right)
$$

Karis doesn't provide any mathematic justification for the additional summation in the denominator, or why we should evaluate $L_i$ by importance sampling GGX, but as [noted by Sebastian Legarde](https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf) (pg. 64) these empirical terms seem to provde the best correction for our split sum approximation for a constant $L_i$.

However, one big problem with this is that the GGX NDF is dependent on $\mathbf{v}$ -- which creates "stretchy" reflections. Karis approximates the NDF as _isotropic_ where $\mathbf{n} = \mathbf{v}$. This is a much larger source of error than the split sum, and is demonstrated by the figure below, which was taken from the previously mentioned [_Moving Frostbite to PBR_ course notes](https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf):

![Comparison showing the error of our isotropic assumption](./isotropic_error.png "Left: Reflections produced by Isotropic approximation. Right: Reference with standard GGX NDF.")

Notice the anisotropic reflections on the right that we lose with the approximation. . While the error is large, it's the price we must pay to be able to perform this sum outside of our render loop when $\mathbf{v}$ is known.

When it comes to storing the pre-filtered environment map, it's important to note once again that we're creating a different map for different roughness levels of our choosing. Since we'll lose high frequency information as roughness increases, we can actually use mip map layers to store the filtered map, with higher roughnesses corresponding to higher mip levels.

Once again we can compute shaders, with a different pass for each roughness/mip level. Here's the loop used in the application code to issue each dispatch call:

```cpp
// width is the width/length of a single face of our cube map
float maxMipLevel = bx::log2(float(width));
for (float mipLevel = 0; mipLevel <= maxMipLevel; ++mipLevel)
{
    uint16_t mipWidth = width >> uint16_t(mipLevel);
    float roughness = mipLevel / maxMipLevel;
    float params[] = { roughness, mipLevel, float(width), 0.0f };
    bgfx::setUniform(u_params, params);
    // Bind the source using a Sampler, so we don't have to perform the cube mapping ourselves
    bgfx::setTexture(0, u_sourceCubeMap, sourceCubeMap);
    // Set the image using a specific mipLevel
    bgfx::setImage(1, filteredCubeMap, uint8_t(mipLevel), bgfx::Access::Write, bgfx::TextureFormat::RGBA16F);
    // Dispatch enough groups to cover the entire _mipped_ face
    bgfx::dispatch(view, preFilteringProgram, mipWidth / threadCount, mipWidth / threadCount, 1);
}
```

Then our computer shader looks like the following:

```glsl
#define THREADS 8
#define NUM_SAMPLES 64u

// u_params.x == roughness
// u_params.y == mipLevel
// u_params.z == imageSize
uniform vec4 u_params;

SAMPLERCUBE(s_source, 0);
IMAGE2D_ARRAY_WR(s_target, rgba16f, 1);

// From Karis, 2014
vec3 prefilterEnvMap(float roughness, vec3 R, float imgSize)
{
    // Isotropic approximation: we lose stretchy reflections :(
    vec3 N = R;
    vec3 V = R;

    vec3 prefilteredColor = vec3_splat(0.0);
    float totalWeight = 0.0;
    for (uint i = 0u; i < NUM_SAMPLES; i++) {
        vec2 Xi = hammersley(i, NUM_SAMPLES);
        vec3 H = importanceSampleGGX(Xi, roughness, N);
        float VoH = dot(V, H);
        float NoH = VoH; // Since N = V in our approximation
        // Use microfacet normal H to find L
        vec3 L = 2.0 * VoH * H - V;
        float NoL = saturate(dot(N, L));
        // Clamp 0 <= NoH <= 1
        NoH = saturate(NoH);

        if (NoL > 0.0) {
            // Based off https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch20.html
            // Typically you'd have the following:
            // float pdf = D_GGX(NoH, roughness) * NoH / (4.0 * VoH);
            // but since V = N => VoH == NoH
            float pdf = D_GGX(NoH, roughness) / 4.0 + 0.001;
            // Solid angle of current sample -- bigger for less likely samples
            float omegaS = 1.0 / (float(NUM_SAMPLES) * pdf);
            // Solid angle of texel
            float omegaP = 4.0 * PI / (6.0 * imgSize * imgSize);
            // Mip level is determined by the ratio of our sample's solid angle to a texel's solid angle
            float mipLevel = max(0.5 * log2(omegaS / omegaP), 0.0);
            prefilteredColor += textureCubeLod(s_source, L, mipLevel).rgb * NoL;
            totalWeight += NoL;
        }
    }
    return prefilteredColor / totalWeight;
}

// Notice the 6 in the Z component of our NUM_THREADS call
// This allows us to index the faces using gl_GlobalInvocationID.z
NUM_THREADS(THREADS, THREADS, 6)
void main()
{
    float mipLevel = u_params.y;
    float imgSize = u_params.z;
    float mipImageSize = imgSize / pow(2.0, mipLevel);
    ivec3 globalId = ivec3(gl_GlobalInvocationID.xyz);

    if (globalId.x >= mipImageSize || globalId.y >= mipImageSize)
    {
        return;
    }

    vec3 R = normalize(toWorldCoords(globalId, mipImageSize));

    // Don't need to integrate for roughness == 0, since it's a perfect reflector
    if (u_params.x == 0.0) {
        vec4 color = textureCubeLod(s_source, R, 0);
        imageStore(s_target, globalId, color);
        return;
    }

    vec3 color = prefilterEnvMap(u_params.x, R, imgSize);
    // We access our target cubemap as a 2D texture array, where z is the face index
    imageStore(s_target, globalId, vec4(color, 1.0));
}
```

I also wrote a [small ShaderToy example](https://www.shadertoy.com/view/WtBSRR) that lets you visualize this for different levels of roughness. The mouse allows you to change the roughness amount. Let's take a look at the resulting output for the [Pisa environment map](http://gl.ict.usc.edu/Data/HighResProbes/):

![Environment map in linear space for different roughness levels.](./pisa_radiance.gif)

Note that this is in linear space and will appear darker than it would in a rendered that properly handles HDR and gamma. The size of the mips has been scaled to match the original image size. Also note the banding is due to GIF encoding, and you shouldn't see anything like that in your output.

### Environment BRDF

Okay, so that's one sum taken care of, now let's look at the second sum. Once again, our goal here is to try and pre-calculate as much of this sum as possible and store it such that we can resolve the integral without having to perform tens/hundreds of texture lookups.

$$
\frac{1}{N} \sum_{k=1}^{N}
    \frac{
        f_s(\mathbf{v}, \mathbf{l}_k) \langle\mathbf{n} \cdot \mathbf{l}_k\rangle
    }{
        \text{pdf}(\mathbf{v}, \mathbf{l}_k)
    }
$$

Let's take a look at it with our specular BRDF and the $\text{pdf}$ subbed in:

$$
\frac{1}{N} \sum_{k=1}^{N}
    \frac{
        D(\mathbf{h}) F(\mathbf{v}\cdot\mathbf{h}) G(\mathbf{v},\mathbf{l},\mathbf{h})
    }{
        4 \langle\mathbf{n}\cdot\mathbf{l_k}\rangle \langle\mathbf{n}\cdot\mathbf{v}\rangle
    }
    \frac{4 \langle \mathbf{v} \cdot \mathbf{h} \rangle
    }{
        D(\mathbf{h}) \langle \mathbf{n} \cdot \mathbf{h} \rangle
    }  \langle\mathbf{n} \cdot \mathbf{l}_k\rangle
= \frac{1}{N} \sum_{k=1}^{N}
    F(\mathbf{v}\cdot\mathbf{h})
    \frac{
        G(\mathbf{v},\mathbf{l},\mathbf{h}) \langle \mathbf{v} \cdot \mathbf{h} \rangle
    }{
        \langle\mathbf{n}\cdot\mathbf{v}\rangle \langle \mathbf{n} \cdot \mathbf{h} \rangle
    }
$$

At this point, we've managed to factor out the NDF but our sum is still dependent on both $\mathbf{v}$, $\text{roughness}$ (through the $G$ term) and $F_0$ (through the Fresnel term). The $F0$ term is particularly inconvenient, because each material will have a potentially different $F0$ term but we don't want to have to store a different LUT for each material. However, if we substitute our Schlick's Fresnel equation and moving thing around a bit:

$$
\begin{aligned}
= &\frac{1}{N} \sum_{k=1}^{N}
    \frac{
        G(\mathbf{v},\mathbf{l}_k,\mathbf{h}) \langle \mathbf{v} \cdot \mathbf{h} \rangle
    }{
        \langle\mathbf{n}\cdot\mathbf{v}\rangle \langle \mathbf{n} \cdot \mathbf{h} \rangle
    }
    F_0(1 - (1 - \langle \mathbf{v} \cdot \mathbf{h} \rangle)^5) + (1 - \langle \mathbf{v} \cdot \mathbf{h} \rangle)^5
\\= &F_0 \left(\frac{1}{N} \sum_{k=1}^{N}
    \frac{
        G(\mathbf{v},\mathbf{l}_k,\mathbf{h}) \langle \mathbf{v} \cdot \mathbf{h} \rangle
    }{
        \langle\mathbf{n}\cdot\mathbf{v}\rangle \langle \mathbf{n} \cdot \mathbf{h} \rangle
    }
    (1 - (1 - \langle \mathbf{v} \cdot \mathbf{h} \rangle)^5)
\right)
\\
& + \left(
        \frac{1}{N} \sum_{k=1}^{N}
        \frac{
            G(\mathbf{v},\mathbf{l}_k,\mathbf{h}) \langle \mathbf{v} \cdot \mathbf{h} \rangle
        }{
            \langle\mathbf{n}\cdot\mathbf{v}\rangle \langle \mathbf{n} \cdot \mathbf{h} \rangle
        }
        (1 - \langle \mathbf{v} \cdot \mathbf{h} \rangle)^5 \right)
\\
= &F_0 f_a + f_b
\end{aligned}
$$

Where $f_a, f_b$ are the _scale_ and the _bias_ terms applied to $F_0$. You may ask yourself what we have accomplished by doing all this, but notice that $f_a, f_b$ are only dependent on $\text{roughness}$ and $\mathbf{n} \cdot \mathbf{v}$. For $\mathbf{n}$ we'll set it to some constant (like the positive z-direction). We can then create a two dimensional LUT where the x-axis is $\langle \mathbf{n} \cdot \mathbf{v} \rangle$ and the y-axis is $\text{roughness}$. The red channel of this LUT will be $f_a$ while the green channel will be $f_b$.

I used a compute shader that just gets run once and the resulting LUT is stored in a RG16F texture. Here's the compute shader code to perform the calculation:

```glsl
#include "bgfx_compute.sh"
#include "pbr_helpers.sh"

#define GROUP_SIZE 256
#define THREADS 16

IMAGE2D_WR(s_target, rg16f, 0);

// Karis 2014
vec2 integrateBRDF(float linearRoughness, float NoV)
{
	vec3 V;
    V.x = sqrt(1.0 - NoV * NoV); // sin
    V.y = 0.0;
    V.z = NoV; // cos

    // N points straight upwards for this integration
    const vec3 N = vec3(0.0, 0.0, 1.0);

    float A = 0.0;
    float B = 0.0;
    const uint numSamples = 1024;

    for (uint i = 0u; i < numSamples; i++) {
        vec2 Xi = hammersley(i, numSamples);
        // Sample microfacet direction
        vec3 H = importanceSampleGGX(Xi, linearRoughness, N);

        // Get the light direction
        vec3 L = 2.0 * dot(V, H) * H - V;

        float NoL = saturate(dot(N, L));
        float NoH = saturate(dot(N, H));
        float VoH = saturate(dot(V, H));

        if (NoL > 0.0) {
            float G = G_Smith(NoV, NoL, linearRoughness);
            float G_Vis = G * VoH / (NoH * NoV);
            float Fc = pow(1.0 - VoH, 5.0);
            A += (1.0 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }

    return vec2(A, B) / float(numSamples);
}


NUM_THREADS(THREADS, THREADS, 1)
void main()
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = vec2(gl_GlobalInvocationID.xy + 1) / vec2(imageSize(s_target).xy);
    float mu = uv.x;
    float a = uv.y;


    // Output to screen
    vec2 res = integrateBRDF(a, mu);

    // Scale and Bias for F0 (as per Karis 2014)
    imageStore(s_target, ivec2(gl_GlobalInvocationID.xy), vec4(res, 0.0, 0.0));
}

```

This shader is a bit simpler than the last, since we don't have to write to a cube map this time. I also wrote a [ShaderToy](https://www.shadertoy.com/view/3lXXDB) for this example as well, which shows you what the output looks like. Note that these outputs will be in RGBA8, so probably aren't suitable for direct use (i.e. you can't use the screen cap as an LUT). The two values smoothly vary and as such we can get away with a fairly small LUT. In my BGFX code I store it in 64x64 texture.

![A high resolution version of the LUT captured from the ShaderToy linked above.](./brdf_lut.png "A high resolution version of the LUT captured from the ShaderToy linked above. The red channel is the scaling factor, and the green channel is the bias (f_a and f_b, respectively)")

## Single Scattering Results

Now that we have a prefiltered environment map and a LUT for most parts of our BRDF, and our diffuse irradiance map, all we have to do is put it together.

$$
\int_{\Omega} f_{\text{brdf}}(\mathbf{v}, \mathbf{l}) L_i(\mathbf{l}) \langle\mathbf{n} \cdot \mathbf{l}\rangle d\mathbf{l}
\approx
\left( F_0 f_a + f_b \right) \text{radiance} + \frac{c_{\text{diff}}}{\pi} \text{irradiance}
$$

Where $f_a, f_b$ are the values stored in our lookup table, $\text{radiance}$ is the prefiltered environment map, \text{irradiance} is the irradiance map, $F_0$ and $c_{\text{diff}}$ are material properties for Specular color and diffuse color respectively. Here's the fragment shader code as well:

```glsl
$input v_position, v_normal, v_tangent, v_bitangent, v_texcoord

#include "../common/common.sh"
#include "pbr_helpers.sh"

#define DIELECTRIC_SPECULAR 0.04
#define BLACK vec3(0.0, 0.0, 0.0)

// Scene
uniform vec4 u_envParams;
#define numEnvLevels u_envParams.x

uniform vec4 u_cameraPos;

// Material
SAMPLER2D(s_diffuseMap, 0);
SAMPLER2D(s_normalMap, 1);
SAMPLER2D(s_metallicRoughnessMap, 2);

// IBL Stuff
SAMPLER2D(s_brdfLUT, 3);
SAMPLERCUBE(s_prefilteredEnv, 4);
SAMPLERCUBE(s_irradiance, 5);

void main()
{
    mat3 tbn = mat3FromCols(
        normalize(v_tangent),
        normalize(v_bitangent),
        normalize(v_normal)
    );
    vec3 normal = texture2D(s_normalMap, v_texcoord).xyz * 2.0 - 1.0;
    normal = normalize(mul(tbn, normal));

    vec3 viewDir = normalize(u_cameraPos.xyz - v_position);
    vec3 lightDir = reflect(-viewDir, normal);
    vec3 H = normalize(lightDir + viewDir);
    float NoV = clamp(dot(normal, viewDir), 1e-5, 1.0);

    // Material properties
    vec4 baseColor = texture2D(s_diffuseMap, v_texcoord);
    vec3 OccRoughMetal = texture2D(s_metallicRoughnessMap, v_texcoord).xyz;
    float occlusion = OccRoughMetal.x;
    float roughness = OccRoughMetal.y;
    float metalness = OccRoughMetal.z;
    // According to GLTF spec
    vec3 F0 = mix(vec3_splat(DIELECTRIC_SPECULAR), baseColor.xyz, metalness);
    float diffuseColor = mix(
        baseColor.rgb * (1.0 - DIELECTRIC_SPECULAR),
        BLACK,
        metalness
    );

    vec2 f_ab = texture2D(s_brdfLUT, vec2(NoV, roughness)).xy;
    // Select appropriate prefiltered environment map based on roughness
    float lodLevel = roughness * numEnvLevels;
    vec3 radiance = textureCubeLod(s_prefilteredEnv, lightDir, lodLevel).xyz;
    vec3 irradiance = textureCubeLod(s_irradiance, normal, 0).xyz;

    vec3 color = (F0 * f_ab.x + f_ab.y) * radiance + (diffuseColor) * irradiance / PI;

    color = occlusion * color * baseColor.w;
    gl_FragColor = vec4(color, baseColor.w);
}
```

You might notice that both the value for `F0` and for `diffuseColor` are derived using the constant `DIELECTRIC_SPECULAR`, which is set to `0.04`. This is from the [GLTF spec](https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#metallic-roughness-material), which uses this value for all dielectrics rather than storing an additional value.

So let's take a look at the results! Here's a render of spheres of increasing roughness (from left to right) that uses the [Pisa courtyard environment map](http://gl.ict.usc.edu/Data/HighResProbes/):

![Rendering of spheres of increasing roughness](./pisa_single_scattering_balls.png "Rendering of spheres of increasing roughness, from left to right. Top row is metal, bottom row is dielectric.")

We can see how as the roughness increases, we're using our blurrier prefiltered environment maps. However, you may also be able to notice that as roughness increases, the metal spheres become darker. One way we can demonstrate this fully is to render the spheres in a totally white environment, where $L_i(\mathbf{l}) = 1$ for all directions $\mathbf{l}$. This is also commonly referred to as a "furnace test":

![Rendering the spheres inside an environment with constant incident radiance](./single_scattering_furnace.png "Rendering the spheres inside an environment with constant L_i = 0.5")

Here the darkening is much more pronounced. About 40% of the energy is being lost, but since our spheres have a "white" albedo, this is violating conservation of energy. Recall that our model is simply a first order approximation which includes only the single scattering events. In reality, light will scatter several times across the surface, as this diagram from [Heitz et al, 2015](https://eheitzresearch.wordpress.com/240-2/) illustrates:

![Diagram illustrating multiple scattering events, from Heitz et al. 2015](./multiple_scattering_diagram.png "Diagram illustrating multiple scattering events, from Heitz et al. 2015")

As the material becomes rougher, the multiscattering events account for a larger proportion of the reflected energy, and our approximation break down. So we need some way of recovering this lost energy if we want to have more plausible results.

## Accounting for Multiple-Scattering

Luckily for us, there's been a significant amount of work done on improving our approximation in order to somehow account for multiple scattering events. [Heitz et al.](https://eheitzresearch.wordpress.com/240-2/) presented their work on modelling the problem as a free-path problem, validating their results by simulating each successive scattering event across triangular meshes using ray tracing. However, their work is too slow for production path tracing, nevermind real time interactive graphics.

In the next few years, [Kulla and Conty](https://blog.selfshadow.com/publications/s2017-shading-course/imageworks/s2017_pbs_imageworks_slides_v2.pdf) presented their own alternative to recovering the energy based on "the furnace test", but it requires additional 2D and 3D LUTs and was meant to be a path tracing solution. Additionally, [Emmanuel Turquin](https://blog.selfshadow.com/publications/turquin/ms_comp_final.pdf) built upon the Heitz paper to model the additional scattering events by noting that Heitz's paper showed the additional scattering events to have lobes resembling smaller versions of the first event's lobe.

Unfortunately none of these solutions are meant for real time, but just this year [Fdez-Agüera](http://www.jcgt.org/published/0008/01/03/paper.pdf) published a paper outlining how we could extend the methods outlined by Karis, without having to introduce any additional LUTs or parameters. The paper provides a full mathematical derivation, but I'll try to provide the highlights.

### Metals

First, let's consider a perfect reflector with no diffuse lobe, where $F = 1$. In this case, the total amount of energy reflected regardless of number of bounces must always be equal to the incident energy (due to conservation of energy):

$$
1 = E_{ss} + E_{ms} => E_{ms} = 1 - E_{ss}
$$

Where $E_{ss}$ is our directional albedo, but with $F = 1$:

$$
E_{ss} = \int_{\Omega} \frac{
    D(\mathbf{h}) G(\mathbf{v},\mathbf{l},\mathbf{h})
}{
    4 \langle\mathbf{n}\cdot\mathbf{l}\rangle \langle\mathbf{n}\cdot\mathbf{v}\rangle
}  \langle\mathbf{n} \cdot \mathbf{l}\rangle d\mathbf{l}
$$

To define $E_{ms}$, we can express it in the same fashion, but with an unknown BRDF:

$$
E_{ms} = \int_{\Omega} f_{ms} \langle\mathbf{n} \cdot \mathbf{l}\rangle d\mathbf{l} = 1 - E_{ss}
$$

So we could effectively add this additional lobe to our reflectance equation:

$$
\begin{aligned}
L_o(\mathbf{v}) &= \int_{\Omega} f_{\text{ss}} + f_{\text{ms}}) L_i(\mathbf{l}) \langle\mathbf{n} \cdot \mathbf{l}\rangle d\mathbf{l}
\\
&= \int_{\Omega} f_{\text{ss}} L_i(\mathbf{l}) \langle\mathbf{n} \cdot \mathbf{l}\rangle d\mathbf{l} + \int_{\Omega} f_{\text{ms}} L_i(\mathbf{l}) \langle\mathbf{n} \cdot \mathbf{l}\rangle d\mathbf{l}
\end{aligned}
$$

We've already discussed the integral on the left, so let's focus exclusively on the second integral. We'll once again use the split sum introduced by Karis, but here Fdez-Agüera makes the assumption that we can consider the secondary scattering events to be mostly diffuse, and therefore use irradiance as a solution to the second integral:

$$
\begin{aligned}
\int_{\Omega} f_{\text{ms}} L_i(\mathbf{l}) \langle\mathbf{n} \cdot \mathbf{l}\rangle d\mathbf{l} &= \int_{\Omega} f_{\text{ms}} \langle\mathbf{n} \cdot \mathbf{l}\rangle d\mathbf{l} \int_{\Omega} \frac{L_i(\mathbf{l})}{\pi} \langle\mathbf{n} \cdot \mathbf{l}\rangle d\mathbf{l}
\\
&= (1 - E_{ss}) \int_{\Omega} \frac{L_i(\mathbf{l})}{\pi} \langle\mathbf{n} \cdot \mathbf{l}\rangle d\mathbf{l}
\end{aligned}
$$

Fdez-Agüera further notes that this approximation fails for narrow lights, such as analytical point or directional lights. Interestingly, [Stephen McAuley](http://advances.realtimerendering.com/s2019/index.htm) made the clever observation that this multiscattering term will only dominate at higher roughnesses, where our radiance map is going to be highly blurred and fairly diffuse. His comparison shows little difference, so if you weren't previously using irradiance you could potentially skip it here as well. However we've already created our irradiance map, so we will be using it again here.

And so, if we use the approximation from earlier for the single scattering event, and bring in this approximation for the multiscattering event, we end up with the following:

$$
L_o(\mathbf{v}) = \left( f_a + f_b \right) \text{radiance} + (1 - E_{ss}) \text{irradiance}
$$

Where $f_a, f_b$ are the scale and bias terms from Karis that we've stored in our LUT.

Recall, however, that we've constrained ourselves to a perfectly reflecting metal here. So in order to extend it to generic metals, we need to re-introduce $F$. Like others, Fdez-Agüera splits $F$ into two terms, $F_{ss}$ and $F_{ms}$ such that:

$$
E = F_{ss} E_{ss} + F_{ms} E_{ms}
$$

However, unlike previously, we cannot simply set $E=1$. Instead, Fdez-Agüera models $F_{ms}$ as a geometric series such that:

$$
E = F_{ss} E_{ss} + \sum_{k=1}^{\inf} F_{ss} E_{ss} (1 - E_{\text{avg}})^k F_{\text{avg}}^k =
$$

Where $F_{\text{avg}}$ and $E_{\text{avg}}$ are defined as:

$$
F_{\text{avg}} = F_0 + \frac{1}{21} (1 - F_0), E_{\text{avg}} = E_{ss}
$$

You'll have to check the papers for the details on this I'm afraid, as I don't want to entirely reproduce the paper in this blog post. The conclusion is that we can therefore write our equation for $L_o$ as the following:

$$
L_o(\mathbf{v}) = \left( F_0 f_a + f_b \right) \text{radiance} + \frac{(F_0 f_a + f_b)F_{\text{avg}}}{1 - F_{\text{avg}}(1 - E_{ss})} (1 - E_{ss}) \text{irradiance}
$$

Let's take a second and look at some renders of metals with this new formulation. For the furnace test, _we should not be able to see the balls at all_. Below is a comparison between the single scattering BRDF and multiple scaterring BRDF inside the furnace:

![Comparison using the spheres in our furnace](./comparison_metals.png "Comparison using the spheres, where for each sphere, single scattering is on the left, multiscattering is on the right")

Sure enough, using our multiple scattering BRDF we can't see the balls at all! The approximation is perfect for constant environments like the furnace, so let's take a look at a less optimal case:

![Render of our metal spheres using the Pisa environment map](./pisa_multiscattering_metals.png "Render of our metal spheres using the Pisa environment map")

The improvement is considerable, as roughness does not darken the sphere at all!

One important thing to mention is that for colored metals, you'll see an increase in saturation at the higher roughnesses not unlike what [Heitz](https://eheitzresearch.wordpress.com/240-2/), [Kulla and Conty](https://blog.selfshadow.com/publications/s2017-shading-course/imageworks/s2017_pbs_imageworks_slides_v2.pdf) and [Hill](https://blog.selfshadow.com/2018/06/04/multi-faceted-part-1/) see. Whether this is desired behaviour is up to you. It appears Turqin's method does _not_ produce these shifts in saturation, so that is worth investigation if increased saturation is to be avoided.

### Dielectrics

So far, we haven't talked about how diffuse fits into this at all. Let's first examine how dielectrics behave in the furnace test for single scattering:

![Render of the dielectric spheres in furnace using single scattering BRDF](./single_scattering_furnace.png "Render of the dielectric spheres in furnace using single scattering BRDF")

As we can see, at lower roughnesses the fresnel effect is too strong, and there is still some brightening at higher roughnesses. Let's go back to the image from earlier and look at the bottom row this time:

$$
E = 1 = F_{ss} E_{ss} + F_{ms} E_{ms} + E_{\text{diffuse}}
$$

This approximation ignores some of the physical reality of diffuse transmission and re-emission, but Fdez-Agüera makes a good argument for why we can keep ignoring them:

> It doesn’t explicitly model the effects of light scattering back and forth between the diffuse and specular interfaces, but since specular reflectance is usually quite low and unsaturated for dielectrics, radiated energy will eventually escape the surface unaltered, so the approximation holds. [(p. 52)](http://www.jcgt.org/published/0008/01/03/paper.pdf)

To extend to non-white dielectrics, we can simply multiply it by the diffuse albedo, $c_{\text{diff}}$.

$$
\begin{aligned}
F_{ss}E_{ss} &= \left( F_0 f_a + f_b \right) \times \text{radiance},
\\
F_{ms}E_{ms} &= \frac{E_{ss} F_{\text{avg}}}{1 - F_{\text{avg}}(1 - E_{ss})} (1 - E_{ss}) \times \text{irradiance},
\\
K_d &= c_{\text{diff}} (1 - F_{ss} E_{ss} + F_{ms} E_{ms}) \times \text{irradiance},
\\
L_o &= F_{ss}E_{ss} + F_{ms}E_{ms} + K_d
\end{aligned}
$$

Let's look at our final results using the same comparison as before, but with dielectrics this time:

![Comparison of our BRDF using dielectric spheres](./comparison_dielectrics.png "Comparison using dielectric spheres, with the single scattering BRDF in use on the left and the multiple scattering BRDF on the right")

Interestingly, I actually get a darker result with multiple scattering than with single scattering. I'm not totally convinced that this isn't some subtle bug in my implementation of the dielectric part of my BRDF. However, the significant excess energy at lower roughnesses that we observe with the single scattering BRDF is not present with our new BRDF. Here's a final render using the Pisa environment, as before, but this time with our new BRDF:

```grid|2|Left: Single scattering. Right: Multiple scattering
![Rendering of spheres of increasing roughness using our single scattering BRDF, from left to right. Top row is metal, bottom rows are dielectric.](./pisa_single_scattering_balls.png)
![Rendering of spheres of increasing roughness using our multiple scattering BRDF, from left to right. Top row is metal, bottom rows are dielectric.](./pisa_multiscattering_balls.png)
```

#### Roughness Dependent Fresnel

In the [Fdez-Agüera](http://www.jcgt.org/published/0008/01/03/paper.pdf) paper's sample GLSL code, the author includes a modification to the Fresnel term that's used with the single scattering BRDF:

```glsl
vec3 Fr = max(vec3_splat(1.0 - roughness), F0) - F0;
vec3 k_S = F0 + Fr * pow(1.0 - NoV, 5.0);
// Typically, simply:
// vec3 FssEss = F_0 * f_ab.x + f_ab.y;
vec3 FssEss = k_S * f_ab.x + f_ab.y;
```

Unfortunately, I haven't been able to track down the origin of this modification, and it's not expanded upon in the paper. It doesn't make much difference when rendering our spheres, especially in the furnace test, but when rendering more complex objects the difference is noticable. To reason about what this term does exactly, I've made a [little desmos graph demo](https://www.desmos.com/calculator/1lvvapokgq). You can adjust the $F_0$ term to see how the modification deviates from the Schlick approximation for different levels of roughness. Interestingly, this makes the ramp starts earlier for _lower_ roughness levels. I can see this being plausible: for rougher surfaces, the proprtion of microfacets contributing to the angle dependent fresnel will decrease. At least that's the best explanation I could come up for it, but if you know the reasoning or source for this modification, please let me know!

Here's a series of renders for you to compare, using the [FlightHelmet](https://github.com/KhronosGroup/glTF-Sample-Models/tree/master/2.0/FlightHelmet) model. The roughness dependent fresnel displays the increase in reflections for smoother surfaces like the leather cap and wooden base.

```grid|3|Left: Flight helmet rendered using single scattering BRDF. Middle: multiscattering BRDF with standed Schlick Fresnel. Right: multiscattering BRDF with roughness dependent Fresnel.
![Flight Helmet rendered using standard single scattering BRDF](./flight_helmet_single_scattering.png)
![Flight Helmet rendered using standard single scattering BRDF](./flight_helmet_multiscattering_standard_fresnel.png)
![Flight Helmet rendered using standard single scattering BRDF](./flight_helmet_multiscattering_roughness_fresnel.png)
```

You can also see the brightening of the metal goggles and base's plaque, but otherwise this model is mostly composed of dielectric materials.

#### GLSL Shader Code

Here's the final shader code, using the BRDF LUT, prefitlered environment map, and irradiance map as uniform arguments:

```glsl
$input v_position, v_normal, v_tangent, v_bitangent, v_texcoord

#include "../common/common.sh"
#include "pbr_helpers.sh"

#define DIELECTRIC_SPECULAR 0.04
#define BLACK vec3(0.0, 0.0, 0.0)

// Scene
uniform vec4 u_envParams;
#define numEnvLevels u_envParams.x

uniform vec4 u_cameraPos;

// Material
SAMPLER2D(s_diffuseMap, 0);
SAMPLER2D(s_normalMap, 1);
SAMPLER2D(s_metallicRoughnessMap, 2);

// IBL Stuff
SAMPLER2D(s_brdfLUT, 3);
SAMPLERCUBE(s_prefilteredEnv, 4);
SAMPLERCUBE(s_irradiance, 5);

void main()
{
    mat3 tbn = mat3FromCols(
        normalize(v_tangent),
        normalize(v_bitangent),
        normalize(v_normal)
    );
    vec3 normal = texture2D(s_normalMap, v_texcoord).xyz * 2.0 - 1.0;
    normal = normalize(mul(tbn, normal));

    vec3 viewDir = normalize(u_cameraPos.xyz - v_position);

    vec4 baseColor = texture2D(s_diffuseMap, v_texcoord);
    vec3 OccRoughMetal = texture2D(s_metallicRoughnessMap, v_texcoord).xyz;

    vec3 lightDir = reflect(-viewDir, normal);
    vec3 H = normalize(lightDir + viewDir);
    float NoV = clamp(dot(normal, viewDir), 1e-5, 1.0);

    float roughness = OccRoughMetal.y;
    float metalness = OccRoughMetal.z;
    float occlusion = OccRoughMetal.x;
    vec3 F0 = mix(vec3_splat(DIELECTRIC_SPECULAR), baseColor.xyz, metalness);

    // IBL stuff starts here
    vec2 f_ab = texture2D(s_brdfLUT, vec2(NoV, roughness)).xy;
    float lodLevel = roughness * numEnvLevels;
    vec3 radiance = textureCubeLod(s_prefilteredEnv, lightDir, lodLevel).xyz;
    vec3 irradiance = textureCubeLod(s_irradiance, normal, 0).xyz;

    vec3 Fr = max(vec3_splat(1.0 - roughness), F0) - F0;
    vec3 k_S = F0 + Fr * pow(1.0 - NoV, 5.0);

    vec3 FssEss = k_S * f_ab.x + f_ab.y;

    float Ess = f_ab.x + f_ab.y;
    float Ems = 1.0 - Ess;
    vec3 F_avg = F0 + (1.0 - F0) / 21.0;
    vec3 Fms = FssEss / (1.0 - Ems * F_avg);

    vec3 k_D = baseColor.xyz * (1 - (FssEss + Fms * Ems));
    vec3 color = FssEss * radiance + (Fms * Ems + k_D) * irradiance;

    color = occlusion * color * baseColor.w;
    gl_FragColor = vec4(color, baseColor.w);
}
```

Notice how little code is actually for calculating our new terms -- most of it is just texture reads and setup that we'd need to do for direct lighting as well.

## Future Work

- The multiscattering BRDF we presented here is only presented for image based lighting. [McAuley](http://advances.realtimerendering.com/s2019/index.htm) goes into some detail into how we can extend it for area lights using [Heitz's](https://eheitzresearch.wordpress.com/415-2/) linearly transformed cosine approach.
- My model doesn't fully support the GLTF material spec. I still need to add better support for Occlusion, Emissive, scaling factors and non-texture uniforms (`baseColor` for instance can be encoded as a single `vec4`). I'm trying to do this bit by bit while implementing other algorithms.
- For our prefiltered environment map, at lower roughness values we are likely to encounter substantial aliasing in the reflections as we can no longer use the cube map's mip map chain as we typically would. I've seen some implementations like Babylon.js actually just use the higher roughness (e.g. blurrier) parts of the map anyway, to reduce aliasing.
- If you have any feedback, please let me know either using the comments below, my [email](mailto:bruno.opsenica@gmail.com) or my [twitter account](https://twitter.com/BruOps)

## Source Code

While I've included most of the shader code used, the entire example is available [here](https://github.com/bruop/bae) as [`04-pbl-ibl`](https://github.com/BruOp/bae/tree/master/examples/04-pbr-ibl). Note however that it uses non-standard GLTF files, that have had their web-compatible textures swapped out with corresponding DDS files that contain mip maps, generated using the `texturec` tool provided as part of BGFX. BGFX does not provide an easy way to invoke the equivalent of `glGenerateMipmap`, so I'm creating a small python script to pre-process the GLTF files, coming soon.

Also I'm going to be re-working some of the application code in the coming days, since it's kind of messy and poorly abstracted.
