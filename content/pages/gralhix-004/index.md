---
layout: default
title: gralhix #004
permalink: /osint/gralhix-004/
---

# gralhix004 | Geolocating Random Islet Image Using Geometry & CUDA GPU Programming 

> 16-08-2026

> NOTE: this is a genuine human work, didnt use LLM generation. 

I'm writing this page as a writeup for this challenge [gralhix 004 made by Sofia Santos | Gralhix](https://gralhix.com/list-of-osint-exercises/osint-exercise-004/).

> You can view, clone and locally try all code files and the final report with all instructions [here at github.](https://github.com/yassa9/geoint/tree/main/gralhix_004)


---

## Task briefing: 

![main](/osint/gralhix-004/main.jpg){: .writeup-img style="width:100%;" }

This is a photo of a resort located on an island.

> a) What is the name of the resort? <br>
> b) What are the coordinates of the island? <br>
> c) In which cardinal direction was the camera facing when the photo was taken?

<div class="project-sep"></div>

In my opinion, solving this challenge with `google lens` is wasting a fun opportunity, so decided to solve it with math and programming.  

---

## a] Metadata

Of course, first thing u look for is the `metadata`. Ran that on my `linux void`: 

```py

> exiftool main.png

File Type                       : WEBP (lossless)
MIME Type                       : image/webp
Image Width                     : 736
Image Height                    : 515

```

As expected, nothing useful here. No EXIF, no GPS, no camera make or model.

---

## b] Building the fingerprint 

![01_00](/osint/gralhix-004/01_00.jpg){: .writeup-img style="width:100%;" }

<div class="project-sep"></div>

U can see from the img, there are 3 landmasses: 

- P0: the islet itself, 
- P1: the right island, 
- P2: the left front island ( having mountain peak )

I couldnt make a correct perspective model of birdview of this image, as clearly the image is taken by a drone and cant estimate the elevation at all (and not found in the metadata).

So I had to estimate that by intuition, I just want the relative distances between the 3 islands and angles of that triangle.

![01_01](/osint/gralhix-004/01_01.jpg){: .writeup-img style="width:35%;" }

I built a small click GUI `01_triangle_gui.py` that records pixel coordinates for each point in order and computes the triangle's geometry.

Since clicking exact centers by eye isn't perfectly precise, I added a `±20% tolerance` band around both values when searching. 

<div class="project-sep"></div>

---

## c] SEARCH

With the fingerprint locked in, the next step is checking every real landmass on Earth against it !

> I used `OpenStreetMap's split land polygon set` as the dataset [land-polygons-split-4326](https://osmdata.openstreetmap.de/data/land-polygons.html), full global coastline vectors in WGS84 which has size of `882 MB`.

I created heuristic filters (all by just intuition and non tangible proofs), spent days (yea full days) tweaking values and tons of trial and error 😭 untill I got this working filters recipe.


### 01] Tropical latitude bounding box

$$ -30° \le latitude \le 30° $$

the islet in the photo reads as tropical, so I decided that anything outside the tropics is thrown out immediately, before doing any expensive geometry work.

> Exactly `141,131` land polygons survive that band filter.


### 02] Local density filter

$$ N_{5\text{km}}(p) \le 10 $$

$ N_{5\text{km}}(p) $ counts how many other centroids fall within 5km of point (p). `Cap is 10`: if an islet has more than 10 neighbors that close, it's sitting in a dense reef field, a crowded coastline or a archipelago clutter, not a small isolated 3-4 island group like the photo shows.

> This dropped candidates down to `51,576`.


### 03] Clustering 

For every surviving point, find every other point within 20km (heuristic, by eye from the image). If it has at least 2 neighbors that close (3 points total), it's a cluster. Points with no cluster of 3+ nearby are dropped, they can't form a triangle at all.

```py

tree = cKDTree(f_coords)
neigh = tree.query_ball_point(
                            f_coords, 
                            CLUSTER_RADIUS_KM / 111.0)
clusters = set(tuple(sorted(n)) for n in neigh if len(n) >= 3)

```

$$ \left|\{q : \text{dist}(p,q) \le 20\,\text{km}\}\right| \ge 3 $$

> That collapses down to `23,500` clusters.


### 04] Generating Triplets

For every cluster, every combination of 3 points inside it becomes a candidate triangle. That's $ C(n, 3) $, which explodes fast for big clusters, for example: a cluster of 60 points already gives `34,220` triples on its own. So each cluster gets capped at 60 points first, sampled by size, not randomly.

$$ \binom{n}{3} = \frac{n(n-1)(n-2)}{6} $$

```py

def stratified_sample(idx_arr, area_arr, cap):
    order = np.argsort(area_arr[idx_arr])
    n_small = cap // 3
    n_large = cap // 3
    n_mid = cap - n_small - n_large
    mid_start = max(0, (len(idx_arr) - n_large - n_mid) // 2)
    keep = np.unique(np.concatenate([
        order[:n_small], 
        order[-n_large:], 
        order[mid_start:mid_start + n_mid],
    ]))
    return idx_arr[keep]

def gen_cluster_triples(idx_arr):
    local = np.array(list(
                itertools.combinations(range(len(idx_arr)), 3)), 
                dtype=np.int64)
    return idx_arr[local]

```

The sampling takes a third small islands, a third large, a third from the middle of the size distribution, instead of the full cluster or a random cut. 

> `23,500` clusters produce `80,690,777` triples total !!


### 05] Matching, on the GPU

I gave every triple one CUDA thread. Each thread sorts its 3 points by land area to pick out P0 (smallest, the resort islet), then uses the winding direction of the other two to assign P1 and P2:

```cpp

long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
if (i >= n_triples) return;

int pos[3] = {0, 1, 2};
for (int a1 = 1; a1 < 3; a1++) 
{
    int key = pos[a1];
    double keyval = a[key];
    int j = a1 - 1;
    while (j >= 0 && a[pos[j]] > keyval) 
    {
        pos[j + 1] = pos[j];
        j--;
    }
    pos[j + 1] = key;
}

```

P1 vs P2 comes from a 2D cross product, no branching on which cluster the triple came from, just the sign:

$$ \text{cross} = x_a y_b - x_b y_a $$ 
$$ P1 = \begin{cases} a & \text{cross} > 0 \\ b & \text{cross} \le 0 \end{cases} $$

Walk from P0 to a, then to b. If cross > 0, that's a left turn (counterclockwise). If cross < 0, it's a right turn (clockwise). It's the same sign trick used to tell if 3 points curve one way or the other.

then angle at P0 and the distance ratio, same formulas as the fingerprint step, computed independently by every thread:

$$ \theta_0 = \arccos\left(\frac{\vec{d_1} \cdot \vec{d_2}}{|\vec{d_1}||\vec{d_2}|}\right), \qquad r = \frac{|\vec{d_1}|}{|\vec{d_2}|} $$

A triple survives if angle, ratio, P0's size, the separation between P0 and P1, and both side lengths all land inside the fingerprint's tolerance windows. Threads that pass write their result into a shared output array using an atomic counter, so two threads finishing at the same time never overwrite each other:

```cpp

if (hit) 
{
    unsigned long long slot = atomicAdd(out_count, 1ULL);
    out_p0[slot] = p0idx;
    out_p1[slot] = p1idx;
    out_p2[slot] = p2idx;
}

```

Now printed in the CLI directly from the kernel:

```py

gpu: NVIDIA GeForce RTX 3050 (sm_86)
vram used: 5169 MB
kernel time: 204.1 ms

```

> `80.7 million` triples go in, one thread each, in parallel. `158,784` pass the mask.


### 06] Dedup

Since same physical triple can get hit by multiple GPU threads if it belonged to more than one overlapping cluster, so raw matches get collapsed by identity first:

```py

seen = set()
uniq = []
for i in range(len(p0_all)):
    key = (p0_all[i], p1_all[i], p2_all[i])
    if key not in seen:
        seen.add(key)
        uniq.append(i)

```

> `8,915` unique triples after dedup.


### 07] The Open Rectangle

![02_00](/osint/gralhix-004/02_00.jpg){: .writeup-img style="width:100%;" }

<div class="project-sep"></div>

Every surviving triple gets one more test: is the space next to it actually open water, like the photo shows ? A rectangle gets built along the P0→P1 edge, on whichever side P2 is not on, then checked against the land dataset for anything else sitting inside it.


```py

width = np.hypot(x1, y1)
u = np.array([x1, y1]) / width
v = np.array([-u[1], u[0]])

# p2 sits on the +v side by construction, 
# so the check goes on -v
length = 2 * width
corners_local = [
    (0, 0), (x1, y1),
    (x1 - v[0]*length, y1 - v[1]*length),
    (-v[0]*length, -v[1]*length),
]

```

If anything other than the 3 candidate islands themselves intersects that rectangle, the candidate is dropped. Land sitting there means it's not the open, unobstructed water the photo actually shows.

> `8,915` unique triples down to `948`.

and below is the map of places of the 948 candidates.

![02_01](/osint/gralhix-004/02_01.jpg){: .writeup-img style="width:100%;" }

<div class="project-sep"></div>

---

## d] Coral Cay Shape Check

In this stage, we look only at P0, the resort islet, and check whether its shape actually looks like a coral cay.

### 1] `Compactness`, how close to a circle the shape is:

`Polsby Popper Score:`
$$ PP = \frac{4\pi \cdot \text{area}}{\text{perimeter}^2} $$

```py

def compactness(row):
    return (4 * np.pi * row.area_km2) / (row.perim_km ** 2 + 1e-12)

```

![03_00](/osint/gralhix-004/03_00.jpg){: .writeup-img style="width:100%;" }

<div class="project-sep"></div>

`1.0` is a perfect circle, lower means a more jagged or elongated outline. Coral cays tend to be round from wave deposition, so anything `< 0.5` gets dropped.

### 2] Micro Cay Halo Check:

```py

def micro_cay_count(gdf, sindex, lon, lat):
    dists_km = nearby.geometry.distance(pt) * 111.0
    mask = (dists_km > 0) 
           & (dists_km <= HALO_KM) 
           & (nearby["area_km2"].values < MICRO_KM2)
    return int(mask.sum())

```

We Count land fragments under 0.05 km² within 1.5km of P0 ( just heuristic ). Real reef systems scatter tiny sandbars around the main island, not just one isolated landmass (I knew that with the hardway 😭). So we need at least 1.

> `213/948` candidates survive both checks.


---

## e] Oval Shape Check

Another geometric filter on P0's own polygon. Fits the minimum rotated rectangle around it and measures two ratios from that box.

```py

def aspect_and_fill(geom):
    mrr = geom.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    s1 = math.hypot(coords[1][0] - coords[0][0], 
                    coords[1][1] - coords[0][1])
    s2 = math.hypot(coords[2][0] - coords[1][0], 
                    coords[2][1] - coords[1][1])
    long_side, short_side = max(s1, s2), min(s1, s2)
    return long_side / short_side, geom.area / mrr.area

```

`Aspect ratio` is long side over short side of that box:

$$ \text{aspect} = \frac{\text{long side}}{\text{short side}} \in [1.05,\ 2.2] $$

Too close to 1.0 and it's basically a perfect circle, not the slightly elongated shape in the photo. Too high are shapes too much elongated more than 2:1.

`Fill ratio` is how much of that bounding box the shape actually fills, and this one has an identity behind it: any ellipse fills precisely $ \pi / 4 $ of its own minimum area bounding rectangle, regardless of how stretched it is.

$$ \frac{\text{area}_{\text{ellipse}}}{\text{area}_{\text{box}}} = \frac{\pi}{4} \approx 0.785 $$

that's the theoretical ceiling for a perfectly smooth oval. Real coral cays aren't perfect ellipses, so the cutoff is set as a heuristic safe fraction of that ceiling:

$$ \text{FILL\_RATIO\_MIN} = 0.75 \times \frac{\pi}{4} \approx 0.589 $$

A shape needs to retain at least 75% of a perfect ellipse's fill to survive. Crescents, rings, and notched coastlines fall well below that, solid rounded cays don't.

> `137/213` candidates survive.


---

## f] NDVI Vegetation Check

We reached the final API phase, I put it at the end, because it is network bound not compute bound.

> We gonna connect to `Earth Search, run by Element84`, a public STAC API that indexes Sentinel-2 imagery hosted on AWS's Open Data program, free, no API key.

You can look at it [https://earth-search.aws.element84.com/v1](https://earth-search.aws.element84.com/v1)

We now check whether P0 is actually vegetated, palm cover, not bare sand or rock. It pulls the most recent low cloud Sentinel-2 scene over the point from a public STAC catalog, samples the red and near infrared bands at that exact pixel.

$$ \text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}} $$

Live vegetation reflects strongly in near infrared and absorbs red light, so healthy palm cover pushes NDVI well above 0, bare sand or open water sits near 0 or negative. 

![04_00](/osint/gralhix-004/04_00.jpg){: .writeup-img style="width:100%;" }

<div class="project-sep"></div>

You can view this image I got from this nice [Geoawesome Blog.](https://geoawesome.com/eo-hub/understanding-aerial-data-normalized-difference-vegetation-index-ndvi/)

`Threshold is set at 0.6`, high enough to require real tree cover, not just scattered units.

> `66/137` survive the NDVI check.

---

## g] Elevation & Mountain Check

![05_00](/osint/gralhix-004/05_00.jpg){: .writeup-img style="width:100%;" }

<div class="project-sep"></div>

Last check before the final reveal. There are two conditions: 

- P0 itself must be low and flat, consistent with a small reef islet, 
- P2 must have real elevated terrain in the direction the camera was actually facing.

The "front" of the shot is the bisector between the bearing to P1 and the bearing to P2:

<div class="project-sep"></div>

$$ \theta(P_0, P_i) = $$ 
$$ \text{atan2}\Big(\sin(\Delta\lambda)\cos\phi_i,\ \cos\phi_0\sin\phi_i - \sin\phi_0\cos\phi_i\cos(\Delta\lambda)\Big) $$

$$ \theta_{\text{front}} = $$
$$ \theta(P_0, P_2) + \frac{\big((\theta(P_0,P_1) - \theta(P_0,P_2) + 180) \bmod 360\big) - 180}{2} $$

That gives one heading, the direction the lens was pointed. From there, a fan of sample points gets swept ±50° around that heading, at radii from 2km out to 20km:

$$ (\text{lat}, \text{lon}) = \Big(\text{lat}_0 + \frac{r\cos\theta}{111},\ \ \text{lon}_0 + \frac{r\sin\theta}{111\cos(\text{lat}_0)}\Big) $$

Every one of those points gets sampled against real `30m Copernicus DEM tiles`. 

> `Copernicus DEM GLO-30`, published by the EU's Copernicus program, hosted as free public Cloud-Optimized GeoTIFFs on AWS Open Data, no account or key needed.

For more info, you can view [https://registry.opendata.aws/copernicus-dem/](https://registry.opendata.aws/copernicus-dem/)

Finally, those two simple heuristic conditions decide survival (yea I know, everything became heuristic haha):

$$ \text{elev}(P_0) \le 50\text{m} $$ 
$$ 100\text{m} \le \max_{\text{arc}}(\text{elev}) \le 500\text{m} $$

<div class="project-sep"></div>

![05_01](/osint/gralhix-004/05_01.jpg){: .writeup-img style="width:100%;" }

<div class="project-sep"></div>

You can see from this abstract graph image, the dashed line is the camera's front bearing, the wedge is the ±50° search arc swept out to 20km for the elevation check.

<div class="project-sep"></div>

> `26/66` survive the elevation check.

<div class="project-sep"></div>
You can see the 26 survivors, all are located in southern Asia, Australia and Oceania, except one near Brazil!

![05_02](/osint/gralhix-004/05_02.jpg){: .writeup-img style="width:100%;" }

<div class="project-sep"></div>

---

## h] Final Report

Finally, last stage, it just makes the final candidates checkable by eye. Each survivor gets its country name via a `point in polygon lookup` against a country boundary file, then a direct Google Maps satellite link for P0, P1, and P2.

Output is a plain HTML table, index, country, three clickable coordinate pairs per row. 

![06_00](/osint/gralhix-004/06_00.jpg){: .writeup-img style="width:100%;" }

<div class="project-sep"></div>

I got this final list, lets check each one by eye.

<div class="project-sep"></div>

Won't go one by one here, but those first 7 are totally off for me.

![06_01](/osint/gralhix-004/06_01.jpg){: .writeup-img style="width:100%;" }

<div class="project-sep"></div>

Till I opened that 8th one in the table of country of Micronesia 😍 (first time to know that a country named Micronesia):

![06_03](/osint/gralhix-004/06_03.jpg){: .writeup-img style="width:100%;" }

<div class="project-sep"></div>

and ensured through P1 and P2:

![06_04](/osint/gralhix-004/06_04.jpg){: .writeup-img style="width:100%;" }

<div class="project-sep"></div>

and that is the solution 🥳 ...

you can view it here on [google maps](https://www.google.com/maps/@7.3633,151.755983,50m/data=!3m1!1e3)

---

## i] FINALLY, ANSWERS ...

```
a) What is the name of the resort? 
``` 

$$ \text{Oan} $$ 

```
b) What are the coordinates of the island?
```

$$7^\circ\,21^\prime\,48.4^{\prime\prime}\,\text{N} \qquad 151^\circ\,45^\prime\,20.7^{\prime\prime}\,\text{E}$$

$$ \text{or} $$ 

$$7.363444^\circ,\ 151.755750^\circ$$

```
c) In which cardinal direction was the 
camera facing when the photo was taken?
```

$$ \because\quad \theta = \text{atan2}\Big(\sin(\Delta\lambda)\cos\phi_1,\ \cos\phi_0\sin\phi_1 - \sin\phi_0\cos\phi_1\cos(\Delta\lambda)\Big) $$

$$ P_0 = (7.3633,\ 151.755983), \quad P_1 = (7.386573,\ 151.739534) $$

$$ \therefore\quad \theta = 324.97^\circ \implies \textbf{NW} $$
