import { useState } from 'react';
import axios from 'axios';
import { FaCloudUploadAlt, FaHeartbeat, FaInfoCircle, FaArrowRight, FaMapMarked, FaExclamation, FaChevronUp, FaChevronDown, FaDownload, FaSpinner } from 'react-icons/fa';
import logo from './assets/logo.jpeg';

const DISEASE_DESCRIPTIONS = {
  'Atelectasis': 'Akciğerin bir kısmının veya tamamının sönmesi (büzüşmesi) durumudur. Genellikle hava yollarının tıkanması sonucu oluşur.',
  'Cardiomegaly': 'Kalbin normal boyutlarından daha büyük olması durumudur (Kalp Büyümesi). Yüksek tansiyon veya kalp yetmezliği belirtisi olabilir.',
  'Effusion': 'Akciğer zarları arasında anormal sıvı birikmesidir (Plevral Efüzyon). Nefes darlığına yol açabilir.',
  'Infiltration': 'Akciğer dokusuna hava yerine sıvı, kan veya iltihap dolmasıdır. Genellikle zatürre veya tüberkülozda görülür.',
  'Mass': 'Akciğerde 3 cm\'den büyük anormal doku büyümesidir. Tümör veya kist olabilir, ileri tetkik gerektirir.',
  'Nodule': 'Akciğerde 3 cm\'den küçük, yuvarlak doku büyümesidir. Genellikle iyi huyludur ancak takip edilmesi gerekir.',
  'Pneumonia': 'Akciğer dokusunun iltihaplanmasıdır (Zatürre). Bakteri, virüs veya mantar kaynaklı olabilir.',
  'Pneumothorax': 'Akciğer ile göğüs duvarı arasına hava kaçmasıdır (Akciğer Sönmesi). Ani nefes darlığı ve ağrı yapar.',
  'Consolidation': 'Akciğerdeki hava keseciklerinin (alveol) sıvı veya iltihapla dolup katılaşmasıdır.',
  'Edema': 'Akciğer dokusunda aşırı sıvı birikmesidir (Ödem). Genellikle kalp yetmezliğine bağlı gelişir.',
  'Emphysema': 'Hava keseciklerinin hasar görüp genişlemesiyle oluşan kronik bir hastalıktır. Genellikle sigara kullanımıyla ilişkilidir.',
  'Fibrosis': 'Akciğer dokusunun kalınlaşması ve sertleşmesidir (Yara dokusu). Akciğerin esnekliğini kaybetmesine neden olur.',
  'Pleural_Thickening': 'Akciğer zarlarının kalınlaşmasıdır. Geçirilmiş enfeksiyonlar veya asbest maruziyeti sonucu oluşabilir.',
  'Hernia': 'Organların (genellikle mide) diyaframdaki bir açıklıktan göğüs boşluğuna kaymasıdır (Fıtık).'
};

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('M');
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState(null);
  const [heatmapUrl, setHeatmapUrl] = useState(null);
  const [expandedDisease, setExpandedDisease] = useState(null);

  const toggleDescription = (diseaseName) => {
    if (expandedDisease === diseaseName) {
      setExpandedDisease(null); 
    } else {
      setExpandedDisease(diseaseName); 
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return alert("Lütfen bir röntgen seçin!");

    setLoading(true);
    setPredictions(null);
    setHeatmapUrl(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('age', age); // Backend yaş bekliyorsa ekleyelim
    formData.append('gender', gender); // Backend cinsiyet bekliyorsa ekleyelim

    try {
      // TAHMİN
      const predRes = await axios.post('http://127.0.0.1:8000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setPredictions(predRes.data.predictions || predRes.data); // Backend yapısına göre değişebilir

      // ISI HARİTASI (Eğer backend ayrı endpoint kullanıyorsa)
      // VEYA predict cevabı içinde base64 dönüyorsa onu kullan:
      if (predRes.data.heatmap_base64) {
         setHeatmapUrl(`data:image/png;base64,${predRes.data.heatmap_base64}`);
      } else {
         // Ayrı endpoint ise:
         const explainRes = await axios.post('http://127.0.0.1:8000/explain', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
            responseType: 'blob' 
         });
         const imageObjectUrl = URL.createObjectURL(explainRes.data);
         setHeatmapUrl(imageObjectUrl);
      }

    } catch (error) {
      console.error("Hata:", error);
      alert("API Hatası! Backend çalışıyor mu? (uvicorn api:app)");
    } finally {
      setLoading(false);
    }
  };

const handleDownloadHeatmap = () => {
    if (!heatmapUrl) return;
    
    const link = document.createElement('a');
    link.href = heatmapUrl;
    link.download = 'ChestXRai-Analiz.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  
  const handleReset = () => {
      setSelectedFile(null);
      setPreviewUrl(null);
      setPredictions(null);
      setHeatmapUrl(null);
      setAge("");
      setGender('M');

      // DÜZELTME: getElementsById -> getElementById
      const fileInput = document.getElementById('file-upload');
      if (fileInput) {
        fileInput.value = '';
      }
  };

  const handleAgeChange = (e) => {
    const value = e.target.value;
    if (value === "") {
      setAge("");
      return;
    }
    const numValue = parseInt(value, 10);
    if(!isNaN(numValue) && numValue >= 0 && numValue <= 150) {
      setAge(value);
    }
  };
 
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if(!file) return; 

    const validTypes = ['image/jpeg', 'image/png', 'image/jpg']; 

    if (!validTypes.includes(file.type)) {
      alert("Hata: sadece JPEG, JPG ve PNG formatındaki resimler kabul edilir")
      e.target.value = null;
      return;
    }

    const maxSize = 5 * 1024 * 1024; 
    if (file.size > maxSize) {
      alert("Hata: Dosya boyutu çok yüksek! Maksimum 5MB yükleyebilirsiniz");
      e.target.value = null;
      return;
    }
    
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setPredictions(null);
    setHeatmapUrl(null);
  };

  return (
    <div className="font-sans text-gray-800 bg-gray-50 min-h-screen">
      <nav className="fixed top-0 w-full bg-white/90 backdrop-blur-md shadow-sm z-50">
        <div className="max-w-7xl mx-auto px-4 h-16 flex justify-between items-center">
          <div className="flex items-center gap-2 cursor-pointer" onClick={() => window.scrollTo(0,0)}>
           <img src={logo} alt="ChestXRai Logo" className="h-12 w-auto object-contain" />
            <span className="font-bold text-2xl text-gray-800">ChestX-R<span className="text-[rgb(70,65,180)]">ai</span></span>
          </div>
          <div className="hidden md:flex space-x-10 font-normal">
            <a href="#analyzer" className="hover:text-blue-600 transition">Analiz</a>
            <a href="#details" className="hover:text-blue-600 transition">Proje Hakkında</a>
            <a href="#about" className="hover:text-blue-600 transition">Biz Kimiz</a>
          </div>
        </div>
      </nav>

      <section id="analyzer" className="pt-32 pb-20 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h1 className="text-5xl font-extrabold text-gray-900 mb-4">
              Yapay Zeka Destekli <span className="text-red-600">Röntgen Analizi</span>
            </h1>
            <p className="text-xl text-gray-500 max-w-2xl mx-auto">
              Saniyeler içinde 14 farklı akciğer hastalığını tespit edin ve xAI teknolojisi ile görsel kanıtları inceleyin.
            </p>
          </div>

          <div className="grid lg:grid-cols-12 gap-8 bg-white p-2 rounded-3xl shadow-2xl border border-gray-100 overflow-hidden">
            {/* SOL PANEL */}
            <div className="lg:col-span-5 p-8 bg-blue-50/50 flex flex-col justify-center">
              <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                <FaCloudUploadAlt className="text-blue-600" /> Görüntü Yükle
              </h2>
              <div className="border-3 border-dashed border-blue-200 rounded-2xl bg-white p-8 text-center hover:border-blue-400 transition-all cursor-pointer relative group h-80 flex flex-col justify-center items-center">
                <input type="file" id="file-upload" accept=".jpg, .jpeg, .png" onChange={handleFileChange} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10" />
                {previewUrl ? (
                  <img src={previewUrl} alt="Preview" className="max-h-full max-w-full rounded-lg shadow-sm object-contain" />
                ) : (
                  <div className="group-hover:scale-105 transition-transform duration-300">
                    <div className="bg-blue-100 p-4 rounded-full inline-block mb-4">
                      <FaCloudUploadAlt className="text-4xl text-blue-600" />
                    </div>
                    <p className="text-gray-500 font-medium">Dosyayı buraya sürükleyin</p>
                  </div>
                )}
              </div>
              <div className="grid grid-cols-2 gap-4 mt-6">
                <div>
                  <label className="text-xs font-bold text-gray-500 uppercase">Yaş</label>
                  <input type="number" value={age} onChange={handleAgeChange} min="0" max="150" className="w-full mt-1 p-3 bg-white border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none" placeholder="45" />
                </div>
                <div>
                  <label className="text-xs font-bold text-gray-500 uppercase">Cinsiyet</label>
                  <select value={gender} onChange={(e) => setGender(e.target.value)} className="w-full mt-1 p-3 bg-white border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none">
                    <option value="M">Erkek</option>
                    <option value="F">Kadın</option>
                    <option value="Other">Diğer</option>
                  </select>
                </div>
              </div>
              <div className="flex gap-3 mt-6">
                {/* ANA BUTON (Analiz Et) */}
                <button 
                  onClick={handleAnalyze} 
                  disabled={loading || !selectedFile}
                  className={`flex-1 py-4 rounded-xl font-bold text-white shadow-lg transition-all transform active:scale-95 ${
                    loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 hover:-translate-y-0.5'
                  }`}
                >
                  {loading ? 'Analiz Ediliyor...' : 'Analizi Başlat'}
                </button>

                {/* SIFIRLA BUTONU */}
                {(selectedFile || predictions) && (
                  <button 
                    onClick={handleReset}
                    className="px-6 py-4 rounded-xl font-bold text-white bg-red-600 hover:bg-red-700 hover:-translate-y-0.5 transition-all active:scale-95 border border-gray-300"
                  >
                    Sıfırla
                  </button>
                )}
              </div>
            </div>

            {/* SAĞ PANEL - SONUÇLAR VE xAI */}
            <div className="lg:col-span-7 p-8 flex flex-col justify-center min-h-[500px]">
              
              {/* DURUM 1: Henüz sonuç yoksa */}
              {!predictions && !loading && (
                <div className="text-center text-gray-400">
                  <FaArrowRight className="text-4xl opacity-20 mx-auto mb-6" />
                  <h3 className="text-xl font-medium">Sonuçlar burada görünecek</h3>
                </div>
              )}

             {/* DURUM 2: Yükleniyorsa */}
              {loading && (
                <div className="flex flex-col items-center justify-center py-12">
                  <FaSpinner className="text-5xl text-blue-600 animate-spin" />
                </div>
              )}

              {/* DURUM 3: Sonuçlar geldiyse */}
              {predictions && (
                <div className="grid md:grid-cols-2 gap-8 animate-fade-in">
                  
                  {/* SOL KOLON: Bulgular Listesi (Akordeon) */}
                  <div>
                    <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                      Bulgular 
                      <span className="text-xs font-normal text-gray-400">(Detay için tıklayın)</span>
                    </h3>
                    
                    <div className="space-y-3">
                      {Object.entries(predictions).slice(0, 4).map(([label, score]) => (
                        <div 
                          key={label} 
                          className={`bg-white rounded-xl border transition-all duration-200 overflow-hidden ${
                            expandedDisease === label ? 'border-blue-400 shadow-md ring-1 ring-blue-100' : 'border-gray-100 shadow-sm hover:border-blue-200'
                          }`}
                        >
                          {/* TIKLANABİLİR BAŞLIK ALANI */}
                          <div 
                            onClick={() => toggleDescription(label)}
                            className="p-3 cursor-pointer"
                          >
                            <div className="flex justify-between items-center mb-2">
                              <span className="font-medium text-gray-700 flex items-center gap-2">
                                {label}
                              </span>
                              
                              <div className="flex items-center gap-2">
                                <span className={`font-bold ${score > 0.5 ? 'text-red-600' : 'text-blue-600'}`}>
                                  {(score * 100).toFixed(1)}%
                                </span>
                                {expandedDisease === label ? 
                                  <FaChevronUp className="text-gray-400 text-xs" /> : 
                                  <FaChevronDown className="text-gray-400 text-xs" />
                                }
                              </div>
                            </div>
                            
                            {/* Progress Bar */}
                            <div className="w-full bg-gray-100 rounded-full h-2">
                              <div 
                                className={`h-full rounded-full transition-all duration-500 ${score > 0.5 ? 'bg-red-500' : 'bg-blue-600'}`} 
                                style={{ width: `${score * 100}%` }}
                              ></div>
                            </div>
                          </div>

                          {/* GİZLİ AÇIKLAMA ALANI */}
                          {expandedDisease === label && (
                            <div className="bg-blue-50 px-3 py-2 text-xs text-gray-600 border-t border-blue-100 flex gap-2 items-start animate-fadeIn">
                              <FaInfoCircle className="text-blue-500 mt-0.5 shrink-0" />
                              <p>{DISEASE_DESCRIPTIONS[label] || "Detaylı açıklama bulunmuyor."}</p>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>

                {/* SAĞ KOLON: xAI Görsel Kanıt */}
                  <div className="flex flex-col h-full">
                    <h3 className="text-lg font-bold text-gray-800 mb-4">xAI Görsel Kanıt</h3>
                    
                    {heatmapUrl ? (
                      <div className="flex flex-col gap-4">
                        {/* Resim Alanı */}
                        <div className="relative group">
                          <img 
                            src={heatmapUrl} 
                            alt="xAI Heatmap" 
                            className="w-full rounded-xl shadow-md border border-gray-200 object-contain max-h-80 bg-black" 
                          />
                          <div className="absolute top-2 right-2 bg-black/60 text-white text-xs px-2 py-1 rounded backdrop-blur-sm">
                            Grad-CAM
                          </div>
                        </div>

                        {/* İndir Butonu */}
                        <button 
                          onClick={handleDownloadHeatmap}
                          className="flex items-center justify-center gap-2 w-full py-3 bg-gray-100 hover:bg-gray-200 text-gray-700 font-semibold rounded-xl transition-all active:scale-95 border border-gray-200"
                        >
                          <FaDownload className="text-blue-600" />
                          Görüntüyü İndir
                        </button>
                      </div>
                    ) : (
                      <div className="h-48 bg-gray-100 rounded-xl flex items-center justify-center text-gray-400 text-sm">
                        Isı haritası yüklenemedi.
                      </div>
                    )}
                  </div>

                </div>
              )}
            </div>

          </div> {/* Grid Kapanışı - EKSİKTİ */}
        </div> {/* Container Kapanışı - EKSİKTİ */}
      </section> {/* Section Kapanışı - EKSİKTİ */}
      
      <section id="details" className="py-24 bg-white border-t border-gray-100">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <h2 className="text-3xl font-bold text-gray-900 mb-10">Proje Hakkında</h2>
          <div className="grid md:grid-cols-3 gap-10">
             <div className="p-6 bg-gray-50 rounded-xl">
               <FaInfoCircle className="text-4xl text-blue-600 mx-auto mb-4"/> 
               <h3 className="font-bold">ResNet-50</h3>
               <p className = "text-gray-600 text-sm leading-relaxed mt-2">
               
               </p>
             </div>
             <div className="p-6 bg-gray-50 rounded-xl">
               <FaMapMarked className="text-4xl text-blue-600 mx-auto mb-4"/> 
               <h3 className="font-bold">Grad-CAM</h3>
               <p className = "text-gray-600 text-sm leading-relaxed mt-2">
                Grad-CAM, yapay zekanın verdiği kararı, görselin hangi bölgesine bakarak verdiğini göstermek için bir ısı haritası oluşturur. Bu x-AI'nın (Explainable AI) temel taşlarından biridir.
               </p>
              </div>
             <div className = "p-6 bg-gray-50 rounded-xl">
               <FaExclamation className="text-4xl text-blue-600 mx-auto mb-4"/> 
               <h3 className="font-bold">Uyarı</h3>
               <p className = "text-gray-600 text-sm leading-relaxed mt-2">
                Bu proje yalnızca eğitim ve akademik araştırma amaçlı geliştirilmiştir. 
                Sunulan sonuçlar kesinlik taşımaz ve profesyonel bir tıbbi teşhis yerine geçmez.
               </p>
             </div>
          </div>
        </div>
      </section>

      <section id="about" className="py-20 bg-gray-50">
        <div className="max-w-4xl mx-auto px-4 text-center">
          <h2 className="text-3xl font-bold text-gray-800 mb-8">Biz Kimiz</h2>
          <div className="flex justify-center gap-10">
            <div className="bg-white p-6 rounded-xl shadow-lg w-64"><h3 className="font-bold text-lg">Türker Kılıç</h3><p className="text-blue-500">AI Engineer</p></div>
            <div className="bg-white p-6 rounded-xl shadow-lg w-64"><h3 className="font-bold text-lg">Ferhat Köknar</h3><p className="text-blue-500">Frontend Developer</p></div>
          </div>
        </div>
      </section>

      <footer className="bg-gray-900 text-white py-8 text-center"><p>&copy; 2025 ChestX-Rai</p></footer>
    </div>
  );
}

export default App;