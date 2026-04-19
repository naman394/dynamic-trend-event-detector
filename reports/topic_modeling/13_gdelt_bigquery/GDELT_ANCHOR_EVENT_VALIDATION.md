# GDELT BigQuery — final anchor validation

We queried **GDELT GKG 2.0** (`gdelt-bq.gdeltv2.gkg_partitioned`) inside each **annotated anchor date window** using the first three **keyword probes** per anchor (OR-matched in `V2Themes` / `V2Persons`, case-insensitive). We also list the strongest **`spike_events.csv`** row in the same calendar window ranked by probe overlap with that anchor.

**Takeaway:** For **black_summer**, **covid_first_au**, and **national_lockdown**, global GDELT volume on the probe vocabulary is clearly elevated in the anchor periods, and our topic pipeline surfaces spikes whose dates and keywords fall in those windows — so the events are **visible both in ABC-derived spikes and in global GDELT news volume**.

---

## `black_summer` — Black Summer bushfires

| Field | Value |
| --- | --- |
| Anchor window | 2019-11-01 … 2020-01-31 |
| GDELT probe terms (first 3) | fire, smoke, evacuation |
| Peak GDELT day (global) | **2020-01-08** |
| Peak daily GKG record count | **28,425** |
| Sum of daily counts in window | **1,624,562** |
| Reference GDELT theme code | `ENV_FIRES` (documentation only; query uses probes) |
| Best `spike_events.csv` in window | topic **139**, **2019-12-23**, z=3.97, probe hits **4** — keywords: *bushfire, bushfires, nsw, fires, victoria* |

## `covid_first_au` — First Australian COVID case

| Field | Value |
| --- | --- |
| Anchor window | 2020-01-20 … 2020-02-05 |
| GDELT probe terms (first 3) | virus, coronavirus, china |
| Peak GDELT day (global) | **2020-01-31** |
| Peak daily GKG record count | **37,254** |
| Sum of daily counts in window | **423,032** |
| Reference GDELT theme code | `HEALTH_PANDEMIC` (documentation only; query uses probes) |
| Best `spike_events.csv` in window | topic **583**, **2020-01-22**, z=4.51, probe hits **3** — keywords: *coronavirus, china, wuhan, chinese, toll* |

## `national_lockdown` — National lockdown

| Field | Value |
| --- | --- |
| Anchor window | 2020-03-01 … 2020-04-15 |
| GDELT probe terms (first 3) | lockdown, restrictions, border |
| Peak GDELT day (global) | **2020-03-18** |
| Peak daily GKG record count | **21,692** |
| Sum of daily counts in window | **518,171** |
| Reference GDELT theme code | `HEALTH_PANDEMIC` (documentation only; query uses probes) |
| Best `spike_events.csv` in window | topic **581**, **2020-03-11**, z=3.90, probe hits **1** — keywords: *coronavirus, italy, lockdown, death, toll* |

## Summary sentence (for reports)

**Detected via GDELT (BigQuery) for all three anchors:** **Black Summer** bushfire season, **first Australian COVID** coverage, and **national lockdown** period each show **strong global GKG article counts** on the anchor keyword probes during the annotated windows, alongside **date-aligned spikes** in our `spike_events.csv` output derived from ABC 2019–2021 topics.
