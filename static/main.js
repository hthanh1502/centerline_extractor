document.getElementById("uploadForm").onsubmit = async function (e) {
    e.preventDefault();
    const fileInput = document.querySelector("input[name='image']");
    const file = fileInput.files[0];
    if (!file) return alert("Vui lòng chọn ảnh.");

    // Hiển thị ảnh chưa xử lý
    document.getElementById("image").src = URL.createObjectURL(file);

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("/process-image", {
        method: "POST",
        body: formData
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({ error: "Lỗi không xác định" }));
        return alert("Lỗi: " + (err.error || "Không thể xử lý"));
    }

    // Hiển thị kết quả
    const data = await res.json();
    const resultBox = document.getElementById("results");
    const resultImage = document.getElementById("resultImage");

    let output = "";
    output += `🔹 Tìm được ${data.segments.length} đoạn centerline\n`;
    data.segments.forEach((seg, i) => {
        output += `Đường ${i + 1}: ${JSON.stringify(seg)}\n`;
    });

    resultBox.textContent = output;
    resultImage.src = data.image_url + "?t=" + new Date().getTime(); // tránh cache
};
