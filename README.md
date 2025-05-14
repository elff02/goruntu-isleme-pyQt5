🍎 Sebze/Meyve Sınıflandırıcı Sistemi
Bu proje, kullanıcıların yüklediği görsellerdeki sebze veya meyveleri yapay zeka kullanarak tanıyan bir masaüstü uygulamasıdır. Kullanıcı dostu arayüzü sayesinde yalnızca birkaç tıklamayla görsel yüklenebilir ve sınıflandırma işlemi gerçekleştirilebilir.
Proje Açıklaması
Sebze/Meyve Sınıflandırıcı, önceden eğitilmiş bir ResNet50 modeliyle çalışır ve kullanıcının yüklediği görselde yer alan nesnenin hangi sebze ya da meyve olduğunu tahmin eder. Sonuçlar, olasılık değerleriyle birlikte ekranda büyük ve anlaşılır bir şekilde sunulur. Uygulama, kullanıcıya sade bir deneyim sunarken, güçlü bir derin öğrenme altyapısıyla yüksek doğruluk sağlar.
Kullanılan Teknolojiler
Proje Python diliyle geliştirilmiştir. Görsel sınıflandırma için PyTorch ve torchvision kütüphaneleri, arayüz tasarımı için PyQt5, görüntü işlemleri için PIL kütüphanesi kullanılmıştır. Ayrıca, sınıf etiketlerini almak için internetten veri çeken küçük bir sistem de mevcuttur.
Kurulum ve Çalıştırma
Projeyi kullanmak isteyenler öncelikle ilgili GitHub reposunu klonlamalıdır. Daha sonra proje dizinine girerek requirements.txt dosyasındaki bağımlılıkları "pip install -r requirements.txt"  komutuyla yüklemelidir. Gerekli kütüphaneler kurulduktan sonra uygulama, terminal üzerinden "python main.py"  komutuyla çalıştırılabilir.
Uygulamanın Kullanımı
Uygulama açıldığında kullanıcıdan bir sebze veya meyve görseli yüklemesi istenir. Daha sonra "Sınıflandır" butonuna tıklanarak işlem başlatılır. Sonuç ekranında en yüksek olasılığa sahip 2 sınıf ve bunlara ait yüzde değerleri kullanıcıya sunulur. Tahmin sonuçları açık ve okunabilir bir şekilde arayüzde gösterilir.
Uygulamanın Özellikleri
Sebze ve meyveleri yüksek doğrulukla sınıflandırabilen bu sistem, internet bağlantısı sayesinde sınıf etiketlerini güncelleyebilir. Arayüz tamamen Türkçe hazırlanmıştır ve herhangi bir teknik bilgiye ihtiyaç duymadan kullanılabilir. Uygulama CPU üzerinde çalıştığı için ek donanım gerektirmez.
Ek Bilgiler
ResNet50 modeli, genel amaçlı bir sınıflandırma modeli olduğundan sınıflar arasında zaman zaman benzerlikler görülebilir. Model ilk çalıştırıldığında bazı dosyaları internet üzerinden indirir, bu nedenle uygulamanın doğru çalışabilmesi için internet bağlantısı gereklidir.
Katkı Sağlamak İsteyenler İçin
Projeye katkı sunmak isteyen geliştiriciler GitHub üzerinden repoyu fork'layabilir, geliştirme için yeni bir branch oluşturabilir ve değişiklikleri commit edip pull request ile katkıda bulunabilirler.
Lisans ve İletişim
Bu proje MIT lisansı ile açık kaynak olarak sunulmuştur. Her türlü soru, öneri ya da iş birliği talebi için GitHub profilimdeki iletişim bilgileri kullanılabilir.
