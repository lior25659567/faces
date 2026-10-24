// The golden vectors in faces-ios/Tests/Fixtures/prompt-golden.json are written by running the
// Python itself (pipeline/prompt_golden.py), and they are the only definition of correct the
// prompt port has. The iOS app asserts against the same file. This is the third implementation of
// those words, so it is held to the same vectors: a prompt that differs by a comma draws a
// different character, and the before/after comparison the whole app exists for is then void.
//
//     node --test
//
// The fixture lives in the sibling iOS checkout. If the two are not side by side, point
// PROMPT_GOLDEN at it. A missing fixture fails the run rather than skipping it, because a test
// that quietly passes when it checked nothing is the one failure mode that matters here.

import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { describe, test } from 'node:test'
import { fileURLToPath } from 'node:url'

import {
  ASK, BODIES, OUTFITS, SIZES, SKIN_SHADES, features, knownOutfit, present, promptFor, seedFor, sizeFor,
} from '../src/prompt.js'

const DEFAULT_GOLDEN = fileURLToPath(new URL('../../../faces-ios/Tests/Fixtures/prompt-golden.json', import.meta.url))
const path = process.env.PROMPT_GOLDEN ?? DEFAULT_GOLDEN

let golden
try {
  golden = JSON.parse(readFileSync(path, 'utf8'))
} catch (problem) {
  throw new Error(
    `Could not read the golden vectors at ${path}: ${problem.message}\n` +
      'Regenerate them with `pipeline/.venv/bin/python pipeline/prompt_golden.py`, ' +
      'or set PROMPT_GOLDEN to where they are.',
  )
}

describe('the pieces of the prompt', () => {
  test('asks the vision model exactly what the Python asks', () => {
    assert.equal(ASK, golden.ask)
  })

  test('colours skin from the same Monk scale', () => {
    assert.deepEqual(SKIN_SHADES, golden.skinShades)
  })

  test('dresses a whole-body avatar in the same outfits', () => {
    assert.deepEqual(Object.fromEntries(OUTFITS), golden.outfits)
  })

  test('describes each body the same way', () => {
    assert.deepEqual(Object.fromEntries(BODIES), golden.bodies)
  })

  test('tries the body words in the Python dictionary order', () => {
    // Not cosmetic: the word is matched by substring, so which one wins for a build the model
    // qualified ("athletic but stocky") is decided entirely by this order.
    assert.deepEqual(BODIES.map(([word]) => word), golden.bodyWords)
  })

  test('asks for the same two output sizes', () => {
    assert.deepEqual(
      { portrait: [SIZES.portrait.width, SIZES.portrait.height], body: [SIZES.body.width, SIZES.body.height] },
      golden.sizes,
    )
    assert.deepEqual(sizeFor(null), SIZES.portrait)
    assert.deepEqual(sizeFor('tee-pants'), SIZES.body)
    // An outfit nobody knows is a portrait, so that every clause agrees with every other rather
    // than asking for a full-length body in clothes nobody specified. index.js refuses such a
    // request before it gets here, which is the honest answer; this is the fallback behind it.
    assert.deepEqual(sizeFor('leotard'), SIZES.portrait)
  })
})

describe('seeds', () => {
  // Names chosen for the shapes of their bytes, not their meaning: the fixture covers an empty
  // name, multi-byte UTF-8, a 200-character name and a lone combining character, because the seed
  // hashes UTF-8 bytes.
  for (const [name, want] of Object.entries(golden.seeds)) {
    test(`${JSON.stringify(name.length > 24 ? `${name.slice(0, 21)}...` : name)} seeds ${want}`, async () => {
      assert.equal(await seedFor(name), want)
    })
  }

  test('stays inside the 31 bits the Python reduces to', async () => {
    for (const value of Object.values(golden.seeds)) {
      assert.ok(Number.isInteger(value) && value >= 0 && value < 2 ** 31)
    }
    assert.equal(await seedFor('p11'), await seedFor('p11'))
    assert.notEqual(await seedFor('p11'), await seedFor('p12'))
  })
})

describe('the facts and the prompt, case by case', () => {
  for (const sample of golden.cases) {
    test(sample.name, () => {
      const { facts, body } = features(sample.seen, sample.outfit)
      assert.equal(facts, sample.facts)
      assert.equal(body, sample.body)
      assert.equal(
        promptFor({ facts, body, outfit: sample.outfit, closeUp: true }),
        sample.promptWithCloseUp,
      )
      assert.equal(
        promptFor({ facts, body, outfit: sample.outfit, closeUp: false }),
        sample.promptWithoutCloseUp,
      )
    })
  }

  test('every case is checked', () => {
    assert.equal(golden.cases.length, 6)
  })
})

describe('the measured sentence stays withdrawn', () => {
  // Its absence is the thing to pin. The measured ratios told the drawing nothing the vision
  // model's word had not already said, and said something false at the heavy end — a severely
  // obese man described exactly as a shredded one. They are still measured and still charted;
  // they are not adjectives for a prompt, and this Worker is never sent them.
  test('the fixture offers no shape phrases to reproduce', () => {
    assert.deepEqual(golden.shapePhrases, [])
  })

  test('features takes only what the model saw and the outfit', () => {
    assert.equal(features.length, 2)
  })

  test('no prompt in the fixture talks about a taper', () => {
    for (const sample of golden.cases) {
      assert.ok(!sample.promptWithCloseUp.includes('taper'))
      assert.ok(!/waist noticeably narrower/.test(sample.promptWithCloseUp))
    }
  })
})

describe('reading the description', () => {
  test("treats the model's words for an absent feature as absent", () => {
    for (const word of ['none', 'None', 'not visible', 'hidden', 'n/a', 'no', 'unknown', '  NONE  ']) {
      assert.equal(present(word), false, `${JSON.stringify(word)} should be absent`)
    }
  })

  test('treats real descriptions as present', () => {
    for (const word of ['light stubble', 'none at all', 'no glasses', 7, '   ']) {
      assert.equal(present(word), true, `${JSON.stringify(word)} should be present`)
    }
  })

  test('drops what cannot be drawn and keeps what can', () => {
    // The three shapes a language model sends that the Mac and the phone read differently. This
    // Worker takes the phone's answer on all three, because it is the phone's avatars these have
    // to match; see the note on asText in prompt.js.
    assert.equal(features({ skin_shade: 7 }, null).facts, 'an adult with a grown-up face; (#825c43) skin')
    assert.equal(
      features({ jewelry: ['gold hoops', 'a thin chain'] }, null).facts,
      'an adult with a grown-up face; gold hoops, a thin chain',
    )
    assert.equal(features({ glasses: true }, null).facts, 'an adult with a grown-up face')
  })

  test('ignores keys inherited from Object.prototype', () => {
    // `seen` is parsed from whatever the vision model wrote, so a key lookup must not reach a
    // function off the prototype and put "function Object()" in front of the image model.
    assert.equal(features({}, null).facts, 'an adult with a grown-up face')
    assert.equal(knownOutfit('constructor'), null)
    assert.equal(knownOutfit('toString'), null)
  })

  test('names only outfits this pipeline has clothes for', () => {
    assert.equal(knownOutfit('tee-pants'), 'tee-pants')
    assert.equal(knownOutfit('tank-shorts'), 'tank-shorts')
    for (const bad of ['', 'Tee-Pants', 'leotard', null, undefined, 42]) {
      assert.equal(knownOutfit(bad), null)
    }
  })
})
