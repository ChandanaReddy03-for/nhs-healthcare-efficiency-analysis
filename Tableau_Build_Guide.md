# Tableau Dashboard Build Guide - NHS Healthcare Efficiency

Estimated build time: 3-4 hours in Tableau Desktop / Tableau Public (2020.2+ for Relationships).
All files are in `data/`. Re-generate any time with `02_Code/nhs_efficiency_pipeline.py`.

> **Important definitions (read first)**
> * **In-clinic wait (`wait_min`)** = minutes between check-in and consultation start, *derived from timestamps*. The supplied `waiting_time` column is winsorised (10% of rows pinned at 100.6 min, 10% at 1.1 min) and understates the mean (40.8 vs 44.1 min). It is kept as `wait_recorded_min`. **Use `wait_min`.**
> * **Booking lead time (`lead_time_days`)** = days from `scheduling_date` to `appointment_date` (the brief's "waiting time" definition).
> * The data contains **one clinic and no specialty column**. Dashboards below use time-of-day, weekday, age band, sex and insurer as the segmentation. If you have a specialty/facility field, add it to `fact_appointments`/`fact_slots` and drop it into the same Filters shelf; no formulas change.

---

## 1. Connect the data

| Data source | File | Grain | Use for |
|---|---|---|---|
| **Appointments** | `fact_appointments.csv` | 1 row per booking (111,488) | waits, lead time, DNA, cancellations, demographics |
| **Slots** | `fact_slots.csv` | 1 row per slot (104,360) | utilisation, capacity loss |
| Daily | `agg_daily.csv` | 1 row per clinic day (2,587) | workload vs wait, overrun |
| Monthly | `agg_monthly.csv` | 1 row per month (120) | trend + 12-month moving average |
| Heatmap | `agg_heatmap_weekday_hour.csv`, `agg_heatmap_month_hour.csv` | pre-aggregated | heatmaps (faster) |
| Segments | `agg_segments.csv` | 1 row per segment | dot plots with 95% CI |
| Scenarios | `scenario_results.csv`, `lp_frontier.csv`, `scenario_grid.csv` | modelling output | optimisation page |

**Data model (Relationships, not joins):** drag `fact_slots` then `fact_appointments`; relate on `slot_id = slot_id` (one slot -> many bookings, because a cancelled slot can be re-booked). Add `patients`-derived fields (already in `fact_appointments`: `insurance`, `age_band`, `sex`).

**Type fixes:** `appointment_date` / `scheduling_date` -> Date; `appointment_datetime` -> Date & Time; `hour`, `dow`, `month`, `year` -> Dimension (discrete); set `weekday` default sort by `dow`.

**Data-source filters (apply on the Slots source):** `in_window = 1` (drops 200 slots after the last data day).

---

## 2. Calculated fields (copy / paste)

```text
// ---------- Appointments source ----------
Avg Wait (min)            = AVG([wait_min])
Median Wait (min)         = MEDIAN([wait_min])
P90 Wait (min)            = PERCENTILE([wait_min], 0.9)
% Wait > 30 min           = AVG([wait_over_30])
% Wait > 60 min           = AVG([wait_over_60])
Avg Lead Time (days)      = AVG([lead_time_days])
Avg Consult (min)         = AVG([duration_min])

DNA Rate                  = SUM([is_dna]) / (SUM([is_attended]) + SUM([is_dna]))
Cancellation Rate         = SUM([is_cancelled]) / SUM([is_resolved])
Cancelled Slot Re-fill %  = AVG([cancel_slot_refilled])           // null for non-cancelled rows

Wait Index (100 = avg)    = AVG([wait_min]) / { FIXED : AVG([wait_min]) } * 100
Winter?                   = IF [is_winter] = 1 THEN "Winter (Dec-Feb)" ELSE "Rest of year" END

// ---------- Slots source ----------
Booked Slot %             = SUM([is_booked_flag]) / COUNT([slot_id])             // brief's definition
Effective Utilisation %   = SUM([is_effective_use]) /
                            SUM(IF [slot_outcome] <> "Unresolved" THEN 1 ELSE 0 END)
DNA % of Slots            = SUM(IF [slot_outcome] = "Did not attend" THEN 1 ELSE 0 END) /
                            SUM(IF [slot_outcome] <> "Unresolved" THEN 1 ELSE 0 END)
Open Slot %               = SUM(IF [slot_outcome] = "Cancelled, not re-filled" OR [slot_outcome] = "Never booked" THEN 1 ELSE 0 END) /
                            SUM(IF [slot_outcome] <> "Unresolved" THEN 1 ELSE 0 END)

// ---------- Daily source ----------
Session Load %            = SUM([workload_min]) / (COUNT([appointment_date]) * 600)
Avg Overrun (min)         = AVG([overrun_min])                       // minutes past 18:00
% Days Finish > 18:30     = SUM(IF [last_end_clock] > 18.5 THEN 1 ELSE 0 END) / COUNT([appointment_date])

// ---------- Parameters ----------
Wait Target (min)         parameter, Integer, default 30, range 10-60, step 5
Wait vs Target            = AVG([wait_min]) - [Wait Target (min)]
Above Target?             = AVG([wait_min]) > [Wait Target (min)]

// ---------- Monthly source ----------
Wait 12M Moving Avg       = WINDOW_AVG(AVG([mean_wait]), -11, 0)     // compute using Table Calculation along Year Month
```

Reference values to sanity-check your build: Avg Wait **44.1 min**, % Wait > 30 **53.8%**, Avg Lead **7.19 days**, DNA Rate **7.14%**, Cancellation Rate **16.46%**, Booked Slot % **89.5%**, Effective Utilisation **83.1%**, Re-fill % **37.9%**, Session Load **96.9%**.

---

## 3. Worksheets

| # | Name | Mark type | Build |
|---|---|---|---|
| W1 | KPI tiles (x6) | Text | One sheet each: Avg Wait, % Wait > 30, DNA Rate, Cancellation Rate, Booked vs Effective Utilisation, Avg Overrun. Add delta vs target using `Wait vs Target`. |
| W2 | Wait by hour of day | Bar | Columns `hour`, Rows `Avg Wait (min)`; colour = `Above Target?` (red/blue); reference line at `Wait Target (min)`. |
| W3 | Heatmap weekday x hour | Square | Columns `hour`, Rows `weekday` (sort by `dow`), colour `Avg Wait (min)` (orange-red diverging, centre on overall mean), label on. |
| W4 | Heatmap month x hour | Square | Columns `hour`, Rows `month_name`; shows winter x late-afternoon hot spot. |
| W5 | Monthly trend | Line | Continuous `year_month`; `Avg Wait (min)` plus dual-axis `Wait 12M Moving Avg` (thick dark-blue line). Shade winter months with a reference band (Dec-Feb). |
| W6 | Seasonality | Bar | Columns `month_name`, Rows `Wait Index (100 = avg)`, colour by `Winter?`. |
| W7 | Slot outcome accounting | Stacked 100% bar | Slots source: Rows `Booked`/`hour`; Columns `% of Total Number of Records`; colour `slot_outcome` (Attended blue, DNA amber, Cancelled-not-refilled red, Never booked grey). |
| W8 | Slots vs time utilisation | Bar | Three bars: Booked Slot %, Effective Utilisation %, Session Load %; reference line at 100%. **Key insight sheet.** |
| W9 | Wait vs workload | Scatter + trend line | Daily source: X `workload_min`, Y `mean_wait`; trend line (linear, show R^2 ~ 0.70); reference line X = 600. |
| W10 | Lead time distribution | Histogram | `lead_time_days` bins of 1; reference line at 22 (IQR fence). |
| W11 | Segments: wait, DNA, cancellation | Dot plot | `agg_segments.csv`: Rows `segment` (filter by `dimension`), Columns `dna_rate`; error bars from `dna_ci95`; reference line = overall DNA rate. Duplicate for `mean_wait`. |
| W12 | Scenario comparison | Bar (2 panels) | `scenario_results.csv`: `mean_wait` and `attended_per_day` by `scenario`; error bars from `*_ci`; reference line = `Wait Target (min)`. |
| W13 | LP frontier | Line + labels | `lp_frontier.csv`: X `attended_per_day`, Y `mean_wait`, label `rho_star`. |

---

## 4. Dashboard layout (desktop, 1400 x 900, fixed)

**Page 1 - Executive overview**
```
+--------------------------------------------------------------------------------------+
| Title + subtitle (clinic, 2015-2024)        | Filters: Year | Month | Weekday | Time block |
+--------------------------------------------------------------------------------------+
| KPI: Avg Wait | % Wait>30 | DNA Rate | Cancel Rate | Utilisation (booked/effective) | Overrun |
+------------------------------------+-------------------------------------------------+
| W2 Wait by hour (bar + target line)| W5 Monthly trend + 12M moving average           |
+------------------------------------+-------------------------------------------------+
| W7 Slot outcome accounting         | Story caption box (see section 6)               |
+------------------------------------+-------------------------------------------------+
```
**Page 2 - Where the inefficiency sits:** W3 heatmap (large), W4 heatmap, W8 slots-vs-time bars, W9 scatter.
**Page 3 - Who and when:** W6 seasonality, W10 lead time, W11 x3 segment dot plots (age band, insurer, sex) with shared "overall" reference line.
**Page 4 - Optimisation:** W13 frontier, W12 scenarios, parameter control `Wait Target (min)`.

**Interactivity**
* Global filters (apply to all sheets using the same source): `year`, `month`, `weekday`, `time_block`, `age_band`, `insurance`, `season`.
* Filter actions: click a bar in W2 (hour) -> filters W3, W5, W7, W9; click a month in W6 -> filters W2, W3.
* Parameter control: `Wait Target (min)` drives reference lines and red/blue colouring everywhere.
* Tooltips: include `Avg Wait`, `P90 Wait`, `% Wait > 30`, visit count, and DNA Rate. Example: `<hour>:00 slots - mean wait <Avg Wait> min (P90 <P90 Wait>). <% Wait > 30> of patients wait over 30 min. n = <visits>`.
* Colour palette (NHS): Blue `#005EB8`, Dark blue `#003087`, Light blue `#41B6E6`, Warm yellow `#FFB81C`, Red `#DA291C`, Green `#009639`, Grey `#768692`.

---

## 5. Publishing checklist
- [ ] Extract (not live) for speed; hide unused fields.
- [ ] Set default filters: Year = All, Time block = All.
- [ ] Add dashboard title tooltip with data window "1 Jan 2015 - 24 Dec 2024".
- [ ] Add a "Data notes" button (floating text) with: wait = timestamp-derived; supplied waiting_time is winsorised; one clinic; no specialty field.
- [ ] If publishing to Tableau Public, drop `patient_id` from `fact_appointments` first (row volume is well inside the limits).

---

## 6. Storytelling captions (numbers verified against the data)

1. **Time of day:** "Waits climb from 16 minutes at 08:00 to 66 minutes at 17:00: about 5.5 minutes more for every hour into the clinic day. Time of day explains 15% of all variation in waiting; weekday, age, sex and insurer explain almost none."
2. **Slots vs clock:** "Only 83% of slots hold an attended patient, yet consultations fill 97% of the 600-minute session, because the average consultation (17.5 min) is longer than the 15-minute slot. The clinic finishes 58 minutes after closing on average."
3. **Winter:** "Winter waits are 11% higher (46.8 vs 42.2 minutes; January +13%). About 70% of that gap is explained by heavier consultation workload on winter days."
4. **Missed slots:** "10% of slots are cancelled and never re-filled and 6.4% are DNAs. Only 38% of cancelled bookings are re-booked."
5. **Targeting:** "DNA rates sit at ~7.1% in every age band, sex and insurer, so reminders work best as a universal intervention."
6. **Optimisation:** "Filling every empty slot without adding capacity raises waits by ~40% (45 -> 63 min). A package of a midday reset, 2 minutes shorter consultations and better re-filling holds throughput while cutting mean wait from 45 to 28 minutes (simulation)."
