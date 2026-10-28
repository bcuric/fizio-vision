# Fizio Vision — prototip

Web-aplikacija (PWA) za mjerenje kuta koljena, kadence hoda i opisne procjene držanja kamerom mobitela.
Procjena poze radi MediaPipe Pose (full model) u pregledniku. Video se ne sprema i ne šalje nikamo,
a mjerenja se spremaju samo na uređaju.

Otvori: https://bcuric.github.io/fizio-vision/

- **Koljeno — opseg pokreta**: fleksija min/max, ROM, ponavljanja; uživo ili iz snimke iz galerije.
- **Hod — kadenca**: kadenca iz medijana trajanja ciklusa, zamasi po nozi.
- **Ramena i glava**: vođeni test s tri ponavljanja (opisno).
- **Spremi mjerenje** uz kut s goniometra; povijest pokazuje razliku i udio unutar ±5°, izvoz u CSV.

Za validaciju je najbolja snimka iz galerije (svaki frame ide kroz model): mobitel vodoravno, bočno,
2–3 m, cijela noga u kadru, odmah zatim izmjeri isti pokret goniometrom.

Prototip za povratne informacije, nije medicinski uređaj i rezultati nisu dijagnoza.
