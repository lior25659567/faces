// Every word that is sent to the two models, and nothing else: this module owns the question put
// to the vision model and the prompt the image model draws from, while index.js owns the calls,
// their shapes and their failures. That is the same split the iOS app makes between
// `AvatarPrompt.swift` and `AvatarService.swift`, for the same reason — the words are the part
// three implementations have to agree on exactly.
//
// This is a port of pipeline/restyle.py, literal down to the punctuation. A person's seed comes
// from their name (`seedFor`) so that their repeat progress photos draw the same character and
// only real bodily change shows between visits; that only holds while the prompt is the same too,
// because the drawing follows the words as much as the seed. A phrase reordered or a comma tidied
// here would make the avatar this Worker draws a different character from the one the Mac and the
// phone draw for the same person, and a before/after comparison then compares two strangers.
// test/prompt.test.js checks every string in here against faces-ios/Tests/Fixtures/prompt-golden.json,
// which is written by running the Python itself and is the only definition of correct this file has.

// What went wrong before this prompt, so nobody reintroduces it:
//   - asked only for a cute avatar, the model drew a generic one: hair on a bald head, light skin
//     for a dark-skinned man, a child's face, neat hair for messy hair;
//   - naming optional features in general ("the same glasses, earrings and head covering") made it
//     add them to people who have none, so only what a person has is named;
//   - a chest-up framing left out tops it found hard to draw, so a portrait shows the waist up;
//   - sending a reference style image as a second picture made it copy that character.
//
// The shade scale, the body words and the ban on brand names are all here rather than in the
// drawing prompt because this is the step that has the photo in front of it.
export const ASK = `You are helping make a cute 3D cartoon avatar that must match this person exactly. Image 1 is their photo; image 2, if there is one, is a close-up of their face.
If there are several people, describe only the one shown in image 2, or else the one whose face is largest.
Describe only what you can see, precisely. Answer with one JSON object only, with these keys:
"age": "child", "teen", "adult" or "senior".
"skin": their skin tone in plain words, e.g. "fair pink", "light warm beige", "medium tan", "warm brown", "deep brown".
"skin_shade": the closest shade on the Monk Skin Tone scale, as a number from 1 (lightest) to 10 (darkest). Warm, orange or red lighting (sunset, red carpets, indoor bulbs, a camera flash) makes skin read lighter and more saturated than it is, and so does bright sun; shade it darker than your first instinct whenever the lighting around the face is warm or bright. If the whites of the eyes are visible, use them as a neutral reference: skin much darker than the eye whites is a high number even if the lighting looks warm or golden.
"face": the face shape and what stands out in it, e.g. "long oval face with high cheekbones and a narrow chin", "round face with full cheeks", "square face with a strong jaw".
"hair": the hair on their head: length, texture (straight, wavy, curly, coily, twists, locs or braids), the exact style (e.g. "short skin fade", "side part swept to the right", "bob with straight bangs", "high ponytail", "big round afro", "messy tousled curls"), the hairline if it is receding or high, and its color. If it is braided or tied, say how many braids, ponytails or buns there are and where they sit (e.g. "one braid over the right shoulder, the rest loose"). If they are bald or the head is shaved, say so plainly. If a head covering hides the hair, say "hidden".
"head_covering": e.g. "black baseball cap worn forwards", "dusty pink hijab", or "none".
"eyes": eye color and shape, e.g. "dark brown almond-shaped eyes", "blue hooded eyes".
"eyebrows": thickness, shape and color.
"facial_hair": e.g. "none", "light stubble", "thick black handlebar mustache", "full bushy white beard", with its exact color, which can differ from the hair (e.g. a reddish-brown beard with dark brown hair), and its length. Mention even light stubble.
"glasses": e.g. "none", "black rectangular glasses", "round tortoiseshell glasses".
"jewelry": visible earrings, piercings, necklaces or watches, or "none".
"makeup": e.g. "none", "red lipstick", "winged eyeliner".
"top": what they wear on the upper body: garment type, color, pattern and collar. If there is a logo or printed text, describe only its shape, color and position (e.g. "a small red rectangular logo on the left chest"); never name the brand or quote the exact words, even if you recognize them.
"bottom": what they wear on the lower body, or "not visible". Describe any logo as above, never by brand name or exact words.
"shoes": their shoes, or "not visible". Describe any logo as above, never by brand name or exact words.
"build": their body size, as one of "slim", "average", "athletic", "stocky", "heavyset", "plus-size", or "not visible". Say "slim" for a thin, narrow body with little muscle, even when it looks lean, toned or in shape; say "athletic" only when muscle is plainly visible, such as defined arms or shoulders, and not merely because someone is slim and flat-stomached. Judge the body you can see, not how fit the person seems. Be precise and honest: this avatar must have their real body.
Do not mention gender, ethnicity, mood, objects they hold, or the background. Do not guess anything you cannot see.`

// The gym outfits of a whole-body avatar. A Map, not an object: the names arrive from an HTTP form,
// and a plain object would answer `OUTFITS["constructor"]` with a function.
export const OUTFITS = new Map([
  ['tee-pants', 'a plain grey T-shirt, black jogger pants and white trainers'],
  ['tank-shorts', 'a plain black athletic tank top, black running shorts and white trainers'],
])

// The Monk Skin Tone scale (Google, free to use), lightest to darkest. The vision model picks a
// shade and the drawing gets its color. Skin measured off the photo was too unreliable for this: a
// sunlit face measured several shades lighter than it is, a backlit one several shades darker.
export const SKIN_SHADES = [
  '#f6ede4', '#f3e7db', '#f7ead0', '#eadaba', '#d7bd96',
  '#a07e56', '#825c43', '#604134', '#3a312a', '#292420',
]

// Body sizes as the vision model names them, and how the image model has to be told: mild words
// such as "plus-size with a full figure" were ignored, and everyone came out slim.
//
// A list of pairs, because the order is part of the behaviour. The Python takes the first of these
// words that appears anywhere in the model's build phrase, and Python dictionaries keep their
// insertion order, so "athletic but stocky" is athletic there — always. Keeping the list in that
// order makes this port reach the same verdict instead of one that depends on key ordering.
//
// None of these says anything about shoulder width. They used to — "athletic" promised broad
// shoulders — and that claim is an impression read off a photo, while shoulder width is something
// body_measure.shape_phrase actually measured. When the two disagreed the prompt said both: a slim
// man the vision model had called athletic was described as having broad shoulders twice over, and
// came back drawn like a bodybuilder.
export const BODIES = [
  ['slim', 'a slim, slender body with thin arms and legs and very little muscle'],
  ['average', 'an average body'],
  ['athletic', 'an athletic body with clearly visible muscle'],
  ['stocky', 'a stocky, solid body with a thick waist'],
  ['heavyset', 'a heavyset body with a big round belly, a thick waist and heavy arms and legs'],
  ['plus-size', 'a plus-size body that is wide and round, with a full round belly, wide hips, thick thighs and full arms'],
]

const STYLE =
  'Style: a polished, cute 3D cartoon avatar, like a premium emoji character: a big rounded head, large expressive eyes ' +
  'with clear colored irises and soft highlights, a small simple nose, smooth soft matte skin, stylized sculpted hair that ' +
  'keeps its real shape, volume and texture, a gentle closed-mouth smile, soft even studio lighting. '

const FRAMING = {
  // A chest-up framing left out tops the model then drew badly, so a portrait is from the waist up.
  portrait:
    'Framing: from the waist up, centered, facing the camera, head upright and level, eyes looking into the camera, ' +
    'arms relaxed, with their clothes clearly visible and empty space above and beside the head. ',
  body:
    'Framing: the whole body from head to feet, standing straight and facing the camera, arms relaxed at the sides, ' +
    'with a slightly larger head. ',
}

const BACKDROP = 'Background: a plain pastel teal studio backdrop, with nothing in their hands and no other people or objects.'

// Workers AI's output sizes: a waist-up portrait, and a taller frame for a whole body.
export const SIZES = {
  portrait: { width: 864, height: 1152 },
  body: { width: 768, height: 1360 },
}

// The words the question above offers the model for an absent feature, plus the ones it reaches for
// anyway. Named features are the whole reason the likeness survives, so "none" must not become a
// part of the prompt saying the person wears none.
const ABSENT = ['none', 'not visible', 'hidden', 'n/a', 'no', 'unknown']

const AGES = new Map([
  ['senior', 'an older adult with a mature face'],
  ['teen', 'a teenager'],
  ['child', 'a child'],
])

// An unreadable or missing age is an adult: the model drew a child's face when it was left to
// decide for itself, and most people scanning a progress photo are adults.
const GROWN_UP = 'an adult with a grown-up face'

/** The outfit to draw, or null for a portrait in the person's own clothes. */
export const knownOutfit = (outfit) => (typeof outfit === 'string' && OUTFITS.has(outfit) ? outfit : null)

/**
 * One value from the vision model's JSON as the text that goes into the prompt, or null to treat
 * the key as absent.
 *
 * The coercion is `PersonSeen`'s forgiving decoder in AvatarPrompt.swift, deliberately, rather
 * than Python's `str()`. Three shapes a language model really does send differ between the Mac
 * and the phone, and this Worker takes the phone's answer on all three, because it is the phone
 * these avatars have to match: an integral float shade (`7.0`) is kept as shade 7 rather than
 * dropped; a list of strings is joined with commas rather than printed in Python's own list
 * notation, brackets and quotes and all; a boolean where words belong is dropped rather than
 * printed, because `{"glasses": true}` has told us nothing to draw and "true" in the prompt is
 * worse than silence.
 */
const asText = (value) => {
  if (typeof value === 'string') return value
  if (typeof value === 'number' && Number.isFinite(value)) return String(value)
  if (Array.isArray(value)) {
    const kept = value.filter((item) => typeof item === 'string' && item.trim() !== '')
    return kept.length ? kept.join(', ') : null
  }
  return null
}

/** Whether a described feature is there at all. */
export const present = (value) => {
  const text = asText(value)
  // Emptiness is tested before trimming and the sentinels after, exactly as the Python does it,
  // which means a whitespace-only value counts as present in both. It is a strange answer for a
  // model to give, but the implementations have to agree on it or the joined sentence differs.
  if (!text) return false
  return !ABSENT.includes(text.trim().toLowerCase())
}

/** Python's `str.isdigit()` for the one place this file depends on it: ASCII digits only. */
const isDigits = (value) => /^[0-9]+$/.test(value)

/**
 * The person's features as one sentence for the drawing, and their body (whole-body avatars only):
 * the vision model's overall impression, as one of BODIES.
 *
 * The measured ratios used to be appended here as a sentence about the taper. They are not any
 * more — see the long note above `measure` in pipeline/body_measure.py, which is where the
 * reasoning lives. In short: over 32 photos that sentence came out the same for a shredded body
 * and a severely obese one, so it told the drawing nothing it did not already have from the word,
 * and told it something false at the heavy end. The ratios are evidence for the person, not
 * adjectives for the prompt, and this Worker is never sent them.
 */
export const features = (seen, outfit) => {
  const described = seen && typeof seen === 'object' && !Array.isArray(seen) ? seen : {}
  const read = (key) => (Object.hasOwn(described, key) ? described[key] : undefined)
  const wanted = knownOutfit(outfit)
  const parts = []

  parts.push(AGES.get((asText(read('age')) ?? '').toLowerCase()) ?? GROWN_UP)

  if (present(read('face'))) parts.push(asText(read('face')))

  let skin = present(read('skin')) ? asText(read('skin')) : ''
  const shade = (asText(read('skin_shade')) ?? '').trim()
  // The hex is given only for a shade the model actually offered, 1 to 10. A number outside that,
  // or anything that is not a plain digit string, leaves the words alone rather than colouring the
  // avatar off the end of the scale. The trim afterwards matters: with no skin words at all the
  // clause is "(#hex) skin", not " (#hex) skin".
  if (isDigits(shade) && Number(shade) >= 1 && Number(shade) <= SKIN_SHADES.length) {
    skin = `${skin} (${SKIN_SHADES[Number(shade) - 1]})`.trim()
  }
  if (skin) parts.push(`${skin} skin`)

  if (present(read('hair'))) parts.push(`hair: ${asText(read('hair'))}`)
  if (present(read('head_covering'))) parts.push(`wearing ${asText(read('head_covering'))}`)

  // The model answers "dark brown almond-shaped eyes" as often as "dark brown almond-shaped", so
  // the key is appended only when the description does not already say it.
  for (const key of ['eyes', 'eyebrows']) {
    if (!present(read(key))) continue
    const text = asText(read(key))
    parts.push(text.toLowerCase().includes(key) ? text : `${text} ${key}`)
  }

  if (present(read('facial_hair'))) {
    const text = asText(read('facial_hair'))
    // Stubble came out as a clean-shaven face unless the drawing was told what stubble looks like,
    // so the one feature that needs it gets its own clause.
    parts.push(
      text.toLowerCase().includes('stubble')
        ? `${text}, drawn as a clearly visible shadow of fine stubble texture over the jaw, chin and upper lip`
        : text,
    )
  }

  for (const key of ['glasses', 'jewelry', 'makeup']) {
    if (present(read(key))) parts.push(asText(read(key)))
  }

  // With an outfit the clothes are being replaced, so describing the real top would fight the
  // instruction that replaces it.
  if (!wanted && present(read('top'))) parts.push(`wearing ${asText(read('top'))}`)

  let body = null
  if (wanted) {
    const build = (asText(read('build')) ?? '').toLowerCase()
    body = BODIES.find(([word]) => build.includes(word))?.[1] ?? null
  }
  return { facts: parts.join('; '), body }
}

/** What the image model is asked to draw. */
export const promptFor = ({ facts, body, outfit, closeUp }) => {
  const wanted = knownOutfit(outfit)
  const shape = body || null
  let text = 'Turn the person in image 1 into a cute 3D avatar character'
  if (shape) text += ` with their real body: ${shape}`
  text += '. '
  if (closeUp) text += 'Image 2 is a close-up of their face. '
  text += STYLE
  // The facts are what keep the likeness; asked only for a cute avatar the model draws a generic
  // one. Without them the prompt at least names which features have to survive.
  text += facts
    ? `It must be unmistakably this person, with exactly these features and nothing added: ${facts}. `
    : 'It must be unmistakably this person, with their own hairstyle, hair color, skin tone, eye color and face. '
  // Said twice on purpose: once as what the character is, once as an instruction. One mention of
  // the body was routinely ignored and the avatar came back slim.
  if (shape) text += `Keep their body exactly this size and shape: ${shape}. `
  if (wanted) text += `Change only their clothes, to ${OUTFITS.get(wanted)}. `
  text += wanted ? FRAMING.body : FRAMING.portrait
  text += BACKDROP
  return text
}

/** The output size: a whole-body avatar needs a taller frame than a waist-up portrait. */
export const sizeFor = (outfit) => (knownOutfit(outfit) ? SIZES.body : SIZES.portrait)

/**
 * A number from a character's name, steady for as long as the name is: Cloudflare's image model
 * draws a recognizably different picture for a different seed, and close to the same picture again
 * for the same one (checked on 3 repeats on the Mac). Reusing it for a person's repeat photos
 * keeps the pose, crop and rendering choices steady from one of their visits to the next, so a
 * later photo differs from an earlier one mostly where they actually do, which is what makes two
 * avatars worth comparing as progress.
 *
 * The Python is `int(sha256(name).hexdigest(), 16) % 2**31` — the whole digest read as one 256-bit
 * big-endian number, then reduced. Matching that needs no big-integer type, and what follows is not
 * an approximation of it: a modulo by a power of two keeps exactly the low bits of a number and
 * discards every higher one, and in a big-endian reading the low 32 bits are the digest's last four
 * bytes, so those four bytes masked to 31 bits are the same value the Python computes, for every
 * input. The golden vectors check it on an empty name, an emoji and a 200-character name.
 */
export const seedFor = async (name) => {
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(String(name ?? ''))))
  const [a, b, c, d] = digest.subarray(-4)
  return (((a << 24) | (b << 16) | (c << 8) | d) >>> 0) & 0x7fffffff
}
