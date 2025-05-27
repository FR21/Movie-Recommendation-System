# Laporan Proyek Machine Learning Terapan - Felix Rafael
## Project Overview
Dalam era digital yang terus berkembang pesat, volume data yang dihasilkan manusia meningkat secara eksponensial setiap harinya, termasuk dalam industri hiburan seperti film. Layanan streaming seperti Netflix, Disney+, Amazon Prime Video, dan lainnya kini menyediakan ribuan hingga puluhan ribu judul yang dapat diakses kapan saja oleh pengguna dari berbagai belahan dunia. Ketersediaan konten yang sangat melimpah ini, meskipun menawarkan banyak pilihan, justru sering kali membuat pengguna mengalami kesulitan dalam menentukan tontonan yang sesuai dengan minat dan preferensi mereka. Untuk itu, sistem rekomendasi (Recommendation System) menjadi alat penting dalam meningkatkan pengalaman pengguna, dengan menyarankan konten yang relevan secara personal.

Netflix, sebagai pelopor dalam penerapan sistem rekomendasi, melaporkan bahwa sekitar 75% aktivitas menonton penggunanya didorong oleh sistem rekomendasi mereka. Menurut **[Gomez (2013)](https://www.wired.com/2013/08/qq-netflix-algorithm)**, sistem ini menganalisis metadata dan perilaku pengguna, termasuk apa yang telah ditonton, dicari, dan dinilai, serta mempertimbangkan faktor-faktor seperti waktu, perangkat, dan lokasi pengguna. Pendekatan ini menunjukkan betapa pentingnya rekomendasi dalam menjaga loyalitas pengguna dan meningkatkan keterlibatan mereka dengan platform. Namun, sistem rekomendasi yang baik memerlukan pendekatan cerdas, seperti pemanfaatan data histori tontonan, ulasan pengguna, hingga fitur berbasis konten seperti genre dan sinopsis film.

Salah satu pendekatan populer dalam membangun sistem rekomendasi adalah _Collaborative Filtering_, yang bekerja berdasarkan kesamaan antar pengguna atau antar item. Namun, pendekatan ini memiliki tantangan seperti _cold start problem_ ketika pengguna atau item baru tidak memiliki cukup data. Untuk mengatasi hal ini, pendekatan _Content-Based Filtering_ dan _Hybrid Models_ dikembangkan guna menggabungkan kekuatan kedua metode. Dalam _Recommender Systems Handbook_, **[Ricci et al. (2015)](https://link.springer.com/book/10.1007/978-1-4899-7637-6)** menjelaskan bahwa sistem rekomendasi berbasis konten memanfaatkan informasi deskriptif dari item, seperti genre dan sinopsis, untuk memberikan rekomendasi yang lebih akurat. 

Dengan berkembangnya teknik pembelajaran mesin dan pemrosesan bahasa alami (Natural Language Processing), sistem rekomendasi dapat dibangun lebih cerdas dan kontekstual. Model seperti _Neural Collaborative Filtering_ (NCF) telah diperkenalkan untuk mengatasi keterbatasan model tradisional dengan memanfaatkan arsitektur jaringan saraf dalam memodelkan interaksi kompleks antara pengguna dan item. **[He et al. (2017)](https://doi.org/10.48550/arXiv.1708.05031)** menunjukkan bahwa pendekatan ini mampu meningkatkan akurasi rekomendasi dengan mempelajari fungsi interaksi yang lebih kompleks dibandingkan dengan metode faktor matriks tradisional.

Proyek ini bertujuan untuk membangun sistem rekomendasi film yang andal dan efisien dengan menggunakan dua pendekatan berbeda, yaitu _Content-Based Filtering_ dan _Collaborative Filtering_. Dengan menggabungkan kedua pendekatan ini, proyek ini tidak hanya berfokus pada konten film itu sendiri, tetapi juga pada perilaku pengguna, sehingga menghasilkan sistem rekomendasi yang lebih adaptif, akurat, dan personal. Sistem ini diharapkan dapat membantu pengguna menemukan film yang sesuai dengan minat mereka secara cepat dan efisien, serta memberikan nilai tambah bagi pengembang aplikasi hiburan digital melalui peningkatan keterlibatan dan retensi pengguna.

## Business Understanding
### Problem Statements
Masalah-masalah utama yang diidentifikasi dalam konteks ini meliputi:
- Terjadinya _overload_ informasi di mana pengguna dihadapkan pada terlalu banyak pilihan sehingga kesulitan menentukan film mana yang paling sesuai dengan minat mereka.
- Kurangnya rekomendasi yang bersifat personal, dapat dilihat dari banyaknya sistem pencarian hanya berdasarkan kategori umum atau popularitas, yang mana belum tentu relevan bagi setiap individu. 
- Adanya masalah _cold start_, yaitu sistem rekomendasi sering kali kesulitan memberikan hasil yang akurat untuk pengguna baru (yang belum memiliki riwayat interaksi) atau film baru (yang belum pernah dinilai).

### Goals
Proyek ini bertujuan untuk membangun sistem rekomendasi film sebagai berikut:
- Mengurangi beban pengguna dalam memilih tontonan dengan menyaring film yang sesuai secara otomatis baik itu berdasarkan genre ataupun _rating_ pengguna lain.
- Dapat memberikan rekomendasi yang relevan dan personal bagi setiap pengguna, serta akurat, efisien, dan dapat diskalakan untuk berbagai jenis pengguna.
- Mengatasi masalah _cold start_, baik dari sisi pengguna maupun _item_, dengan menggabungkan pendekatan berbasis konten dan interaksi.

### Solution Statement
Untuk mencapai tujuan di atas, proyek ini mengadopsi dua pendekatan utama dalam pengembangan sistem rekomendasi, yaitu:
1. **`_Content-Based Filtering_`**:
Pendekatan ini merekomendasikan film kepada pengguna berdasarkan kemiripan antara _item_, bukan antar pengguna. Sistem ini:
    - Menggunakan _Natural Language Processing_ (NLP) untuk mengekstraksi fitur dari deskripsi film.
    - Menerapkan teknik _TF-IDF_ (Term Frequency-Inverse Document Frequency) untuk merepresentasikan genre film dalam bentuk vektor.
    - Menghitung _cosine similarity_ antar vektor deskripsi untuk mengidentifikasi film yang mirip dengan yang pernah disukai pengguna.
    - Cocok untuk mengatasi _cold start_ pada film baru, karena hanya bergantung pada metadata film, bukan histori interaksi pengguna.

2. **`_Collaborative Filtering_`**:
Pendekatan ini merekomendasikan film berdasarkan pola interaksi antar pengguna dan _item_. Sistem ini:
    - Menggunakan arsitektur _Neural Collaborative Filtering_ (NCF) dan _RecommenderNet_ yang memodelkan interaksi pengguna-item dengan jaringan saraf.
    - Menggeneralisasi teknik _matrix factorization_ melalui pembelajaran fitur laten pengguna dan item menggunakan _embedding layers_.
    - Memungkinkan pemodelan interaksi non-linear yang lebih kompleks dan akurat daripada _collaborative filtering tradisional_.
    - Efektif dalam menangkap preferensi implisit pengguna dari data interaksi, seperti _rating_.
