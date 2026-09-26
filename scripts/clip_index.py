import csv
import os
import re
from dataclasses import dataclass, field

PAT = re.compile(
    r"^(?P<cid>\d{8}_[A-Z]{3}at[A-Z]{3}_P\d_\d{4}_\d__a\d)"
    r"(?P<labels>(?:_[A-Z]{3}_[A-Za-z]+)+)\.mp4$")

COMPOUND = {
    "whiteandlightblue": ["white", "lightblue"],
    "blackandpurple": ["black", "purple"],
    "blueandred": ["blue", "red"],
}
HUE = {"red": 0.00, "darkred": 0.99, "lightgreen": 0.42, "orange": 0.07, "yellow": 0.15, "gold": 0.13, "cream": 0.11,
       "green": 0.35, "teal": 0.48, "lightblue": 0.55, "blue": 0.60,
       "darkblue": 0.63, "navy": 0.63, "purple": 0.78}
ACHROMATIC = {"white", "black", "grey", "gray", "silver"}

@dataclass
class Clip:
    cid: str
    path: str
    event_id: str
    away: str = ""
    home: str = ""
    label: str = ""
    colours: dict = field(default_factory=dict)

    @property
    def complete(self):
        return len(self.colours) == 2

    def components(self, team):
        c = self.colours.get(team, "")
        return COMPOUND.get(c, [c] if c else [])

    def is_achromatic(self, team):
        comp = self.components(team)
        return bool(comp) and all(c in ACHROMATIC for c in comp)

    def hues(self, team):
        return [HUE[c] for c in self.components(team) if c in HUE]

def load_index(base, clips_dir=None, manifest=None):
    clips_dir = clips_dir or os.path.join(base, "clips", "clips")
    manifest = manifest or os.path.join(base, "manifest.csv")
    man = {r["event_id"]: r for r in
           csv.DictReader(open(manifest, encoding="utf-8-sig"))}

    idx = {}
    for f in sorted(os.listdir(clips_dir)):
        if not f.endswith(".mp4"):
            continue
        m = PAT.match(f)
        if not m:
            cid = f[:-4]
            idx[cid] = Clip(cid, os.path.join(clips_dir, f), cid.split("__a")[0])
            continue
        cid = m.group("cid")
        toks = m.group("labels").strip("_").split("_")
        colours = {toks[i]: toks[i + 1].lower() for i in range(0, len(toks) - 1, 2)}
        eid = cid.split("__a")[0]
        row = man.get(eid, {})
        idx[cid] = Clip(cid, os.path.join(clips_dir, f), eid,
                        row.get("away_team", ""), row.get("home_team", ""),
                        row.get("label_touched_last", ""), colours)
    return idx

def cached(base, stage, cid):
    p = os.path.join(base, "cache", "cache", stage, cid + ".npz")
    return p if os.path.exists(p) else None

if __name__ == "__main__":
    b = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    idx = load_index(b)
    full = [c for c in idx.values() if c.complete]
    both = [c for c in full if cached(b, "detect", c.cid) and cached(b, "pose", c.cid)]
    print(f"{len(idx)} clips indexed, {len(full)} with both colours")
    print(f"{len(both)} of those have detect+pose cached")
    miss = [c.cid for c in full if not cached(b, "detect", c.cid)]
    print(f"{len(miss)} awaiting cache" + (f" (e.g. {miss[0]})" if miss else ""))
    n_ach = sum(1 for c in full if c.is_achromatic(c.away) or c.is_achromatic(c.home))
    print(f"{n_ach} clips where at least one team is white/black (no hue)")
    print(f"{sum(1 for c in full if c.is_achromatic(c.away) and c.is_achromatic(c.home))}"
          f" clips where BOTH teams are achromatic -- hue cannot separate these")
