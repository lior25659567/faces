# Faces

A cute 3D avatar of you whose head follows the pointer, made from a photo.

## How it works

1. **Restyle, by hand.** The Gemini app turns a photo of you plus `reference/style.png` into one character still. Steps and prompt: [pipeline/restyle-prompt.md](pipeline/restyle-prompt.md).
2. **Frames, locally.** [LivePortrait](https://github.com/KwaiVGI/LivePortrait) re-renders the still's head at a 7 × 5 grid of angles, gives each frame back the still's fine detail, and packs the 35 results into one atlas image. Nothing is uploaded.
3. **Effect.** `<FaceFollow />` lays one cell of the atlas over the head in the still and dissolves from cell to cell as the pointer moves.

## Run it

```bash
npm install
npm run setup                    # once: Python env, LivePortrait and weights, about 1.8 GB in pipeline/
npm run frames -- input/me.png   # about 2 minutes on an M1; writes src/faces/me/
npm run dev                      # shows every character in src/faces/
```

Needs macOS on Apple silicon, Node 22, [uv](https://docs.astral.sh/uv/) and Xcode's command line tools (`xcode-select --install`), which build the small Swift program that finds faces with Apple's Vision framework.

## Make a character from the page

Under `npm run dev` the page has two buttons, **Use camera** and **Upload image**. Either one sends the picture to the frame generator and switches to the new character: a coarse 5 × 3 preview after about a minute, then the full 7 × 5 grid a couple of minutes later, in place.

- With **3D style** off, a picture is animated as it is and never leaves this computer.
- With **3D style** on, it is first drawn as a cute 3D avatar of the person. That switch is locked until it is set up (see [The 3D style switch](#the-3d-style-switch)).
- The picture is saved in `input/`.
- A camera photo is always saved as the character `camera`, so a new one replaces the last. An uploaded image is named after its file.
- **Upload image** takes any picture format this Mac can open: JPEG, PNG, WebP, HEIC straight from an iPhone, TIFF, GIF.
- The buttons are dev-only. They talk to a small endpoint in [pipeline/frames-api.js](pipeline/frames-api.js); a built site only shows the characters already in `src/faces/`.

## Whole-body gym avatar

The **Avatar** picker next to the buttons switches from a portrait to a whole-body avatar in gym clothes, as a starting point:

- **Men's:** a plain grey T-shirt, black jogger pants and white trainers.
- **Women's:** a plain black athletic tank top, black running shorts and white trainers.

With an outfit picked, **Use camera** becomes **Scan body**: prop the screen upright, start the 10-second timer and step back until you fit from head to feet. **Upload image** takes a head-to-feet photo instead. Either way the avatar keeps your face, hair, skin tone and body size, standing straight with arms at the sides, and only the clothes change.

- It always goes through the restyle, so it needs the Cloudflare key and counts against the daily allowance (next section).
- A scan is saved as the character `body`, so it never replaces your `camera` portrait. An upload is named after its file.
- The head turns as in a portrait. In a whole-body picture the head is small, so the effect is smaller too.
- The outfits are written out in `OUTFITS` in [pipeline/restyle.py](pipeline/restyle.py). A new one also needs its name in [pipeline/frames-api.js](pipeline/frames-api.js) and in the picker in [face-source.jsx](src/components/face-source/face-source.jsx).

### Body shape, measured

For a whole-body photo, [pipeline/body_measure.py](pipeline/body_measure.py) measures the torso's actual shape instead of leaving it to the vision model's one-word guess ("athletic", "stocky"): Apple Vision finds the shoulder and hip joints, and the shoulder-to-waist and waist-to-hip proportions are read from the photo's own silhouette at those heights, as a share of the person's own shoulder width. The two numbers (`waist_to_shoulder`, `hip_to_shoulder`) describe their real taper, not a bucket; they sharpen the drawn avatar, and they are the figures worth charting for progress over time, since they hold steady for the same photo while the drawn avatar does not.

What this needs to work:
- **Arms a little away from the sides**, as the capture dialog already asks for. Arms pressed flush against the torso can hide the waist from the measurement, which then falls back to the vision model's word alone.
- **Fitted or form-fitting clothing.** A dress, a long coat, or anything baggy hides the body's outline from Vision's own pose model, not only from this measurement — there is no way around this for any camera-based method. When it happens, the log says so and the avatar is still drawn, from the description's impression alone.

Two more things make repeat photos of the same person comparable rather than each landing on an unrelated-looking avatar:
- **A steady seed.** The drawing call is seeded from a hash of the character's name (`--name`, passed by the dev server), so the same person's photos render with the same posing and rendering choices from one visit to the next.
- **A steady description.** The vision model runs at temperature 0, so the same photo reads back the same hairstyle, jewelry and build each time, instead of a different word for the same earrings.

Even so, two separate calls to a hosted image model are not guaranteed pixel-identical — that is a real limit of the service, not something this project can close. The measured numbers are exact; the avatar is a consistent likeness, not a measuring instrument. An app that wants real progress tracking should chart the numbers and use the avatar as the picture that goes with them, not the evidence itself.

## The 3D style switch

The switch turns each camera photo or upload into a cute 3D avatar, in the spirit of Apple's Memoji but not a copy: a big rounded head, large eyes, smooth skin and sculpted hair, on a teal studio backdrop. A portrait shows the waist up, in the person's own clothes. Cloudflare's free daily allowance covers it; no card is needed.

1. Create a free Cloudflare account.
2. In the dashboard, open Workers AI and choose **Use REST API**. It shows your account ID and creates an API token.
3. Copy `.env.example` to `.env` and fill in the two values. The dev server restarts and the switch unlocks.

How it keeps the likeness, in [pipeline/restyle.py](pipeline/restyle.py):

1. **Find the face, on this computer**, with Apple's Vision framework, for a close-up of it. A photo too dark to read is turned away here, before anything is sent.
2. **Describe.** A vision model on Cloudflare writes down the face shape, skin tone on the [Monk Skin Tone scale](https://skintone.google), hairstyle and hairline, eye color and shape, eyebrows, facial hair, glasses, jewelry, makeup, clothes and body size.
3. **Draw.** FLUX.2 klein on Cloudflare draws the avatar from the photo, a close-up of the face, and that list. Only what the person has is named, because naming a feature in general, such as glasses, made the model add it to everyone.

What to expect:

- **Allowance.** 10,000 "neurons" a day, free, shared by everything the account runs. An avatar uses about 145: about 130 for the drawing and 14 for the description, so about 65 a day. Setting `RESTYLE_MODEL=@cf/black-forest-labs/flux-2-klein-9b` in `.env` uses about 1,400 each, so about 7 a day.
- **Time.** About 25 seconds for the avatar, then the frames as before.
- **Privacy.** With the switch on, the picture is uploaded to Cloudflare twice, once to be described and once to be drawn. With it off, nothing leaves this computer.
- **Likeness.** Tested on 20 people of different ages, skin tones, hair types and builds, including glasses, beards, a hijab, a cap and plus-size bodies. Hairstyle, hair and skin color, glasses, facial hair, jewelry, clothes with their logos and patterns, and body size come through. The face is a cute cartoon of theirs, not a portrait, and light stubble often comes out too faint.
- **Photos.** Use one where the face is in clear light and facing the camera. Each avatar comes out a little different, so upload again if one misses.
- **The clay look.** For the exact look of `reference/style.png`, restyle by hand (step 1 at the top) and upload the result with the switch off.

## Tuning the frames

`npm run frames -- --help` lists every option. The ones worth knowing:

| Option | Default | What it changes |
| --- | --- | --- |
| `--yaw`, `--pitch` | 18, 12 | How far the head turns at the edges of the viewport, in degrees. |
| `--eyes-x`, `--eyes-y` | 8, 18 | Extra eye travel on top of the head turn. Sideways travel over about 12 smears large cartoon eyes. |
| `--steps` | 7 | Frames across. Must be odd. The atlas has a fixed size, so more frames means smaller, softer ones. |
| `--rows` | 5 | Frames down. Must be odd. By default, as many as turn the head by the same amount per frame as across. |
| `--scale` | 2.5 | Width of the animated square, in face widths. |
| `--face x0,y0,x1,y1` | found automatically | The face's box in pixels of the source, forehead to chin and cheek to cheek. Use it if the face isn't found. |

## Use it in another project

Copy `src/components/face-follow/` and `src/faces/me/`, then:

```jsx
import FaceFollow from './components/face-follow/face-follow.jsx'
import base from './faces/me/base.webp'
import atlas from './faces/me/atlas.webp'
import manifest from './faces/me/manifest.json'

const Hero = () => (
  <FaceFollow face={{ base, atlas, ...manifest }} alt="Lior as a 3D character" className="hero__figure" />
)
```

Give it a width; the height follows the still. `manifest.backdrop` lists the still's backdrop colors from top to bottom, for pages that continue it (see `backdropGradient` in `src/app.jsx`). The head stays still when the visitor asks for reduced motion.

## License note

Every part the pipeline runs allows commercial use, for example in an App Store app:

- **Finding faces and measuring bodies:** Apple's Vision framework, part of macOS and iOS.
- **Drawing the avatar:** FLUX.2 klein 4B, under Apache 2.0. Keep it: `RESTYLE_MODEL=@cf/black-forest-labs/flux-2-klein-9b` is under Black Forest Labs' non-commercial license.
- **Describing the person:** Google's Gemma 4, under Apache 2.0.
- **Turning the head:** LivePortrait, code and weights under MIT.

Older installs also have the InsightFace face detector in `pipeline/vendor/LivePortrait/pretrained_weights/insightface/`. It is for non-commercial use and nothing uses it any more, so it can be deleted.
