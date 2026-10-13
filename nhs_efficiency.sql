-- =====================================================================================
-- NHS Healthcare Efficiency Analysis - SQL layer (SQLite; tested on 3.45)
-- Tables: appointments <- cleaned_appointments.csv, patients <- patients.csv, slots <- slots.csv
-- Order: 0 load -> 1 profiling -> 2 views -> 3 KPIs -> 4 analysis -> 5 Tableau views
-- For PostgreSQL / SQL Server swap strftime()/julianday() (PG equivalents noted inline).
-- =====================================================================================

-- ---------- 0. LOAD (the brief's Step 2 queries) -----------------------------------
SELECT * FROM appointments LIMIT 5;
SELECT * FROM patients     LIMIT 5;
SELECT * FROM slots        LIMIT 5;

-- ---------- 1. DATA-QUALITY CHECKS --------------------------------------------------
-- 1a. duplicate keys (expect 0 rows each)
SELECT appointment_id, COUNT(*) c FROM appointments GROUP BY appointment_id HAVING c > 1;
SELECT patient_id,     COUNT(*) c FROM patients     GROUP BY patient_id     HAVING c > 1;
SELECT slot_id,        COUNT(*) c FROM slots        GROUP BY slot_id        HAVING c > 1;

-- 1b. orphan foreign keys (expect 0)
SELECT COUNT(*) AS orphan_slots    FROM appointments a LEFT JOIN slots    s ON s.slot_id    = a.slot_id    WHERE s.slot_id    IS NULL;
SELECT COUNT(*) AS orphan_patients FROM appointments a LEFT JOIN patients p ON p.patient_id = a.patient_id WHERE p.patient_id IS NULL;

-- 1c. missing timing data is structural: NULL exactly when the visit was not attended
SELECT status, COUNT(*) AS rows_n, SUM(waiting_time IS NULL) AS null_wait, SUM(appointment_duration IS NULL) AS null_duration
FROM appointments GROUP BY status ORDER BY rows_n DESC;

-- 1d. winsorisation check: share of attended rows sitting exactly on the recorded min / max
SELECT ROUND(100.0*SUM(waiting_time >= (SELECT MAX(waiting_time) FROM appointments) - 0.05)/COUNT(*),2) AS pct_wait_at_ceiling,
       ROUND(100.0*SUM(waiting_time <= (SELECT MIN(waiting_time) FROM appointments) + 0.05)/COUNT(*),2) AS pct_wait_at_floor,
       ROUND(100.0*SUM(appointment_duration >= (SELECT MAX(appointment_duration) FROM appointments) - 0.05)/COUNT(*),2) AS pct_duration_at_ceiling
FROM appointments WHERE status = 'attended';

-- ---------- 2. CLEANING / STANDARDISATION VIEWS -------------------------------------
DROP VIEW IF EXISTS v_appointments_clean;
CREATE VIEW v_appointments_clean AS
SELECT a.appointment_id, a.slot_id, a.patient_id, a.scheduling_date, a.appointment_date,
       a.appointment_date || ' ' || a.appointment_time                       AS appointment_datetime,
       CAST(strftime('%Y', a.appointment_date) AS INT)                       AS year,         -- PG: EXTRACT(YEAR FROM ..)
       CAST(strftime('%m', a.appointment_date) AS INT)                       AS month,
       strftime('%Y-%m', a.appointment_date)                                 AS year_month,
       CAST(substr(a.appointment_time,1,2) AS INT)                           AS hour,
       CASE WHEN CAST(strftime('%m', a.appointment_date) AS INT) IN (12,1,2) THEN 'Winter'
            WHEN CAST(strftime('%m', a.appointment_date) AS INT) IN (3,4,5)  THEN 'Spring'
            WHEN CAST(strftime('%m', a.appointment_date) AS INT) IN (6,7,8)  THEN 'Summer' ELSE 'Autumn' END AS season,
       lower(trim(a.status))                                                 AS status,
       a.sex, a.age, a.age_group, p.insurance,
       -- booking lead time (brief: scheduling_date -> appointment_date). PG: appointment_date - scheduling_date
       CAST(julianday(a.appointment_date) - julianday(a.scheduling_date) AS INT) AS lead_time_days,
       -- timestamp-derived (UNCAPPED) in-clinic minutes. PG: EXTRACT(EPOCH FROM (start_time::time - check_in_time::time))/60
       (strftime('%s','2000-01-01 '||a.start_time) - strftime('%s','2000-01-01 '||a.check_in_time))/60.0 AS wait_min,
       (strftime('%s','2000-01-01 '||a.end_time)   - strftime('%s','2000-01-01 '||a.start_time))/60.0     AS duration_min,
       (strftime('%s','2000-01-01 '||a.end_time)   - strftime('%s','2000-01-01 00:00:00'))/3600.0          AS end_hour,
       a.waiting_time AS wait_recorded_min, a.appointment_duration AS duration_recorded_min
FROM appointments a JOIN patients p ON p.patient_id = a.patient_id;

-- one row per slot with its final outcome (utilisation + capacity-loss accounting)
DROP VIEW IF EXISTS v_slot_outcome;
CREATE VIEW v_slot_outcome AS
WITH per_slot AS (
  SELECT slot_id, COUNT(*) AS n_bookings,
         MAX(status = 'attended') AS has_att, MAX(status = 'did not attend') AS has_dna,
         MAX(status IN ('scheduled','unknown')) AS has_unres
  FROM appointments GROUP BY slot_id)
SELECT s.slot_id, s.appointment_date, s.appointment_time,
       CAST(substr(s.appointment_time,1,2) AS INT) AS hour,
       strftime('%Y-%m', s.appointment_date)       AS year_month,
       CASE WHEN s.is_available IN (0,'0','False','false') THEN 1 ELSE 0 END AS is_booked,   -- brief: booked = not available
       CASE WHEN s.appointment_date > (SELECT MAX(appointment_date) FROM appointments) THEN 'Outside data window'
            WHEN ps.has_att   = 1   THEN 'Attended'
            WHEN ps.has_dna   = 1   THEN 'Did not attend'
            WHEN ps.has_unres = 1   THEN 'Unresolved'
            WHEN ps.slot_id IS NULL THEN 'Never booked'
            ELSE 'Cancelled, not re-filled' END AS slot_outcome
FROM slots s LEFT JOIN per_slot ps ON ps.slot_id = s.slot_id;

-- ---------- 3. HEADLINE KPIs ----------------------------------------------------------
-- 3a. in-clinic waiting (attended visits)
SELECT ROUND(AVG(wait_min),1) AS mean_wait_min, ROUND(AVG(wait_recorded_min),1) AS mean_wait_recorded_capped,
       ROUND(AVG(duration_min),2) AS mean_consult_min,
       ROUND(100.0*AVG(wait_min > 30),1) AS pct_wait_gt_30, ROUND(100.0*AVG(wait_min > 60),1) AS pct_wait_gt_60
FROM v_appointments_clean WHERE status = 'attended';

-- 3b. booking lead time (days)
SELECT ROUND(AVG(lead_time_days),2) AS mean_lead_days, MIN(lead_time_days) AS min_days, MAX(lead_time_days) AS max_days FROM v_appointments_clean;

-- 3c. slot utilisation: brief formula (booked / total x 100) vs effective (attended) vs losses
SELECT ROUND(100.0*SUM(is_booked)/COUNT(*),2)                                                  AS booked_slot_pct,
       ROUND(100.0*SUM(slot_outcome='Attended')/SUM(slot_outcome<>'Unresolved'),2)             AS effective_pct_attended,
       ROUND(100.0*SUM(slot_outcome='Did not attend')/SUM(slot_outcome<>'Unresolved'),2)       AS dna_pct_of_slots,
       ROUND(100.0*SUM(slot_outcome IN ('Cancelled, not re-filled','Never booked'))/SUM(slot_outcome<>'Unresolved'),2) AS open_pct_of_slots
FROM v_slot_outcome WHERE slot_outcome <> 'Outside data window';

-- 3d. missed-appointment KPIs (DNA = DNA / (attended + DNA); cancellation = cancelled / all resolved)
SELECT ROUND(100.0*SUM(status='did not attend')/SUM(status IN ('attended','did not attend')),2) AS dna_rate_pct,
       ROUND(100.0*SUM(status='cancelled')/SUM(status IN ('attended','did not attend','cancelled')),2) AS cancel_rate_pct
FROM v_appointments_clean;

-- 3e. cancellation re-fill: cancelled bookings whose slot was later used by another booking
SELECT COUNT(*) AS cancelled_bookings,
       SUM(EXISTS (SELECT 1 FROM appointments b WHERE b.slot_id = c.slot_id AND b.status <> 'cancelled')) AS refilled,
       ROUND(100.0*SUM(EXISTS (SELECT 1 FROM appointments b WHERE b.slot_id = c.slot_id AND b.status <> 'cancelled'))/COUNT(*),1) AS refill_pct
FROM appointments c WHERE c.status = 'cancelled';

-- ---------- 4. ANALYSIS ---------------------------------------------------------------
-- 4a. waiting by hour of day (dominant pattern)
SELECT hour, COUNT(*) AS visits, ROUND(AVG(wait_min),1) AS mean_wait, ROUND(AVG(duration_min),1) AS mean_consult
FROM v_appointments_clean WHERE status='attended' GROUP BY hour ORDER BY hour;

-- 4b. seasonality: monthly mean wait and index vs overall average (100 = average)
WITH m AS (SELECT month, AVG(wait_min) AS w FROM v_appointments_clean WHERE status='attended' GROUP BY month),
     o AS (SELECT AVG(wait_min) AS w FROM v_appointments_clean WHERE status='attended')
SELECT m.month, ROUND(m.w,1) AS mean_wait, ROUND(100.0*m.w/o.w,1) AS index_vs_avg FROM m, o ORDER BY m.month;

-- 4c. winter vs rest of year
SELECT (season = 'Winter') AS is_winter, COUNT(*) AS visits, ROUND(AVG(wait_min),2) AS mean_wait, ROUND(AVG(lead_time_days),2) AS mean_lead_days
FROM v_appointments_clean WHERE status='attended' GROUP BY 1;

-- 4d. 12-month moving average of monthly waiting (window function)
WITH ym AS (SELECT year_month, AVG(wait_min) AS w FROM v_appointments_clean WHERE status='attended' GROUP BY year_month)
SELECT year_month, ROUND(w,2) AS mean_wait,
       ROUND(AVG(w) OVER (ORDER BY year_month ROWS BETWEEN 11 PRECEDING AND CURRENT ROW),2) AS wait_ma12
FROM ym ORDER BY year_month;

-- 4e. demographics: waiting and DNA by age group / insurance / sex
SELECT age_group, COUNT(*) AS visits, ROUND(AVG(wait_min),1) AS mean_wait FROM v_appointments_clean WHERE status='attended' GROUP BY age_group ORDER BY age_group;
SELECT insurance, ROUND(AVG(CASE WHEN status='attended' THEN wait_min END),1) AS mean_wait,
       ROUND(100.0*SUM(status='did not attend')/SUM(status IN ('attended','did not attend')),2) AS dna_rate_pct
FROM v_appointments_clean GROUP BY insurance ORDER BY mean_wait DESC;
SELECT sex, COUNT(*) AS visits, ROUND(AVG(wait_min),1) AS mean_wait FROM v_appointments_clean WHERE status='attended' GROUP BY sex;

-- 4f. utilisation by hour: booked vs attended vs DNA vs open
SELECT hour, ROUND(100.0*SUM(is_booked)/COUNT(*),1) AS booked_pct,
       ROUND(100.0*SUM(slot_outcome='Attended')/COUNT(*),1) AS attended_pct,
       ROUND(100.0*SUM(slot_outcome='Did not attend')/COUNT(*),1) AS dna_pct,
       ROUND(100.0*SUM(slot_outcome IN ('Cancelled, not re-filled','Never booked'))/COUNT(*),1) AS open_pct
FROM v_slot_outcome WHERE slot_outcome NOT IN ('Outside data window','Unresolved') GROUP BY hour ORDER BY hour;

-- 4g. daily capacity: consultation minutes vs the 600-min session, and finish time
DROP VIEW IF EXISTS v_daily;
CREATE VIEW v_daily AS
SELECT appointment_date,
       SUM(status='attended') AS attended, SUM(status='did not attend') AS dna, SUM(status='cancelled') AS cancelled,
       ROUND(SUM(CASE WHEN status='attended' THEN duration_min END),1) AS workload_min,
       ROUND(AVG(CASE WHEN status='attended' THEN wait_min END),2)     AS mean_wait,
       ROUND(MAX(CASE WHEN status='attended' THEN end_hour END),2)     AS last_end_hour
FROM v_appointments_clean GROUP BY appointment_date;

SELECT ROUND(AVG(workload_min),1) AS avg_consult_min_per_day, ROUND(100*AVG(workload_min)/600.0,1) AS pct_of_600min_session,
       ROUND(AVG(last_end_hour),2) AS avg_finish_hour, ROUND(100.0*AVG(last_end_hour > 18.5),1) AS pct_days_finish_after_1830
FROM v_daily WHERE attended > 0;

-- 4h. heavier days -> longer waits (load quartiles)
WITH q AS (SELECT *, NTILE(4) OVER (ORDER BY workload_min) AS load_quartile FROM v_daily WHERE attended > 0)
SELECT load_quartile, COUNT(*) AS days, ROUND(AVG(workload_min),0) AS avg_workload_min, ROUND(AVG(mean_wait),1) AS avg_mean_wait
FROM q GROUP BY load_quartile ORDER BY load_quartile;

-- ---------- 5. TABLEAU-READY VIEW --------------------------------------------------------
DROP VIEW IF EXISTS v_tableau_monthly;
CREATE VIEW v_tableau_monthly AS
SELECT year_month, year, month, COUNT(*) AS appointments,
       SUM(status='attended') AS attended, SUM(status='did not attend') AS dna, SUM(status='cancelled') AS cancelled,
       AVG(CASE WHEN status='attended' THEN wait_min END) AS mean_wait, AVG(lead_time_days) AS mean_lead_days,
       1.0*SUM(status='did not attend')/NULLIF(SUM(status IN ('attended','did not attend')),0) AS dna_rate,
       1.0*SUM(status='cancelled')/NULLIF(SUM(status IN ('attended','did not attend','cancelled')),0) AS cancel_rate
FROM v_appointments_clean GROUP BY year_month, year, month;
SELECT * FROM v_tableau_monthly ORDER BY year_month LIMIT 12;
