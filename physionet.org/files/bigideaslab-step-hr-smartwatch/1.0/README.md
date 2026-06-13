# README

## Data (deidentified_data.csv)

The observations are synced, so that each row represents a single point in time. There are 10 columns:

- `ECG`: heart rate (HR) reported by Bittium Faros 180 ECG Patch (beats per minute, bpm) (~1000 Hz)
- `Apple Watch`: HR reported by Apple Watch (bpm) (variable sampling frequency)
- `Empatica`: HR reported by Empatica E4 (bpm) (~1Hz)
- `Garmin`: HR reported by Garmin (bpm) (variable sampling frequency)
- `Fitbit`: HR reported by Fitbit (bpm) (variable sampling frequency)
- `Miband`: HR reported by Xiaomi Miband (bpm) (variable sampling frequency)
- `Biovotion`: HR reported by Biovotion Everion (bpm) (~1Hz)
- `ID`: internal study ID
- `Skin Tone`: skin tone of participant (standard Fitzpatrick skin tone scale, 1-6)
- `Activity`: the activity a participant was participating in (Rest, Activity, Breathe, Type). For a complete description of these conditions, please see the associated publication.

## Protocol (protocol.pdf)

The protocol.pdf file documents the study protocol.
