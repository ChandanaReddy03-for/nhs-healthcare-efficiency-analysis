<<<<<<< HEAD
# NHS Healthcare Efficiency Analysis - end-to-end project pack

| Folder | Contents |
|---|---|
| `01_Report/` | Written report (see the Claude doc; export to Word/PDF from there) |
| `02_Code/` | `nhs_efficiency_pipeline.py` - clean -> transform -> analyse -> LP + Monte Carlo -> charts -> Tableau exports; `requirements.txt` |
| `03_SQL/` | `nhs_efficiency.sql` - profiling, cleaning views, KPI + analysis queries, Tableau view (SQLite-tested; PG notes inline) |
| `04_Tableau/` | `Tableau_Build_Guide.md` (data model, calculated fields, sheets, layout, captions) and `data/` (11 Tableau-ready CSVs) |
| `05_Charts/` | 13 PNG charts used in the report |
| `results.json` | Every statistic quoted in the report (machine-readable) |

## Reproduce
```bash
pip install -r 02_Code/requirements.txt
python 02_Code/nhs_efficiency_pipeline.py --data-dir <folder with the 3 CSVs> --out-dir .
```
Runtime ~2-3 minutes. Business assumptions (wait target 30 min, session 08:00-18:00, 40 x 15-min slots) are constants at the top of the script.

## Headline findings
* In-clinic wait (timestamp-derived) averages **44.1 min**, rising from **15.7 min at 08:00 to 66.0 min at 17:00**.
* Slots look **83%** used, but consultations fill **97%** of the 600-min session (17.5-min consults in 15-min slots); clinic finishes **58 min late** on average.
* **10.1%** of slots are cancelled and never re-filled, **6.4%** are DNAs; only **38%** of cancellations are re-booked.
* Winter waits are **+11%**; ~70% of that is explained by heavier workload. No significant decade trend.
* DNA/cancellation rates are flat across age, sex and insurer (DNA model AUC 0.50).
* Filling gaps without capacity raises waits ~40%; recommended package (reset block + 2-min shorter consults + re-fill + reminders) holds ~33 patients/day and cuts mean wait to ~28 min (simulation).

## Data caveats
* No specialty or facility field exists; segmentation is by time, weekday, age, sex, insurer.
* Supplied `waiting_time` / `appointment_duration` are winsorised at the 10th/90th percentile; analysis uses timestamp-derived values.
* Tableau workbook (.twbx) is not included - build from the guide.
=======
# nhs-healthcare-efficiency-analysis
End-to-end NHS healthcare efficiency analysis using Python, SQL, statistical modelling, optimisation, and Tableau-ready data to investigate patient waiting times, appointment utilization, cancellations, and clinic capacity.
>>>>>>> bd2804db7cf1dd53c801cdcbda2bf3b06954c8e4
