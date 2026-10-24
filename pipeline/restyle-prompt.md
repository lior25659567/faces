# Restyle prompt

Turns one photo of you into the character still. It is a single image, so the free Gemini app is enough.

1. Open the Gemini app and start a chat that can create images.
2. Attach two images in this order: your photo, then `reference/style.png`.
3. Paste the prompt below and send. Regenerate until the likeness is right.
4. Save the result, then give it to **Upload image** on the demo page (`npm run dev`). From the terminal instead: save it as `input/me.png` and run `npm run frames -- input/me.png`.

Use a photo where you face the camera in even light, without sunglasses.

## Prompt

```text
Recreate the person in the first image as a stylized 3D character, rendered in exactly the style of the second image.

Likeness: keep them recognizable. Match their hairstyle and hair color, skin tone, face shape, eyebrows, and any facial hair, glasses or other distinctive features, simplified into the style rather than copied photographically.

Style, taken from the second image: smooth matte clay-like skin with a soft blush on the cheeks, a slightly oversized head on a slender neck, a small simple nose, large round white eyes with small solid black pupils, thick sculpted eyebrows, a small understated mouth, and hair built from chunky sculpted clumps. They wear a fuzzy fleece hoodie with white drawstrings over a white T-shirt, like the reference. Soft, even studio lighting, as in a high-end 3D animation render.

Pose: facing the camera straight on, head upright and level, eyes looking directly into the camera, mouth closed with a relaxed, friendly expression. Arms down by the sides, hands below chest height and well away from the face.

Framing: vertical 9:16, waist up, centered, with clear empty space above the head and on both sides of it.

Background: the same plain pastel teal studio backdrop as the second image. No props, no text, no other people.
```

For the reference's shrug, replace the last sentence of the pose paragraph with: `Both forearms raised with palms open and facing up in a light shrug, hands at chest height and clear of the face.`

## What the frame generator needs from the still

The generator re-renders a square around the face, about 2.5 times the face's size, and leaves the rest of the image untouched.

- **Face front-on, eyes on the camera.** This still becomes the straight-ahead frame; every other angle is rendered relative to it.
- **Whole head in frame, with backdrop around it.** The head needs room to turn inside that square.
- **Hands and props away from the head.** Anything inside the square is warped along with the head.
- **Plain backdrop.** The backdrop inside the square is warped too, and a flat color hides it.
