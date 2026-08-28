"""
make_vlm_neutral.py — strip text leakage from the VLM battery.

Round 1 showed the GPT-written contexts leak place identity through cultural
color (no-image W_rr = 0.63). This rewrites every context as a NEUTRAL template
with zero local cues: the city bridge can then only enter through the image.
Distractor/filler words come from a generic pool, unique across episodes.
"""
import json, argparse

D_POOL = ["courier", "argument", "cyclist", "umbrella", "siren", "puddle", "pigeons",
          "scaffolding", "megaphone", "balloon", "stroller", "whistle", "ladder",
          "barricade", "drone", "skateboard", "traffic", "camera", "backpack",
          "raincoat", "scooter", "crowd", "queue", "vendor", "microphone", "banner",
          "confetti", "firetruck", "ambulance", "helicopter", "protest", "parade",
          "puppet", "accordion", "juggler", "magician", "spill", "alarm", "glitter",
          "trombone", "unicycle", "mascot", "flashmob", "bullhorn", "tarp", "crane",
          "jackhammer", "generator", "forklift", "billboard", "turnstile", "kiosk",
          "fountain", "bench", "railing", "lamppost", "dumpster", "hydrant",
          "mailbox", "signpost", "awning", "gutter", "manhole", "pothole", "curb",
          "crosswalk", "overpass", "stairwell", "escalator", "elevator", "cone",
          "podium", "speaker", "amplifier", "tripod", "easel", "sketchpad",
          "notebook", "pencil", "eraser", "wallet", "keychain", "bottle", "thermos",
          "sandwich", "napkin", "receipt", "ticket", "brochure", "map", "compass",
          "watch", "glove", "scarf", "button", "zipper", "shoelace", "pocket",
          "helmet", "vest", "clipboard", "whiteboard", "marker", "stapler", "folder",
          "envelope", "stamp", "parcel", "crate", "trolley", "wheelbarrow", "shovel",
          "broom", "bucket", "mop", "sponge", "towel", "blanket", "pillow", "curtain",
          "mirror", "frame", "poster", "sticker", "magnet", "candle", "lantern",
          "flashlight", "battery", "charger", "cable", "headphones", "radio",
          "speakerphone", "printer", "scanner", "keyboard", "mouse", "monitor",
          "projector", "screen", "remote", "antenna", "satellite", "telescope"]

TEMPLATE = ("{p} arrived early and waited near the structure shown in the photo. "
            "A {d1} nearly knocked over a cyclist during a loud {d2}, and {p} "
            "stepped back, clutching a {f1}. After a while the commotion faded "
            "and {p} kept waiting.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/benchmarks/battery_vlm.json")
    ap.add_argument("--out", default="data/benchmarks/battery_vlm2.json")
    args = ap.parse_args()

    battery = json.load(open(args.inp))
    pool = iter(D_POOL)
    out = []
    for ep in battery:
        person = ep["probe_question"].split("should ")[-1].split(" use")[0].strip() \
            if "should " in ep["probe_question"] else "Alex"
        words = [w for _, w in zip(range(3), pool)]
        if len(words) < 3:
            break
        d1, d2, f1 = words
        ctx = TEMPLATE.format(p=person, d1=d1, d2=d2, f1=f1)
        city = [it["concept"] for it in ep["items"] if it["label"] == "load_bearing"][0]
        items = [{"concept": city, "label": "load_bearing", "role": "silent visual bridge"},
                 {"concept": d1, "label": "distractor", "role": "vivid in-text"},
                 {"concept": d2, "label": "distractor", "role": "vivid in-text"},
                 {"concept": f1, "label": "filler", "role": "neutral in-text"}]
        out.append({"context": ctx, "image": ep["image"],
                    "probe_question": ep["probe_question"], "answer": ep["answer"],
                    "landmark": ep["landmark"], "items": items})
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"saved {len(out)} neutral episodes / {sum(len(e['items']) for e in out)} items -> {args.out}")


if __name__ == "__main__":
    main()
