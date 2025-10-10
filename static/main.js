// main.js
document.getElementById("uploadForm").onsubmit = async function (e) {
    e.preventDefault();
    const fileInput = document.querySelector("input[name='image']");
    const file = fileInput.files[0];
    if (!file) return alert("Vui lòng chọn ảnh.");

    // Hiển thị ảnh chưa xử lý
    document.getElementById("image").src = URL.createObjectURL(file);

    // Thêm phần tử loading vào giao diện
    let loading = document.getElementById("loading");
    if (!loading) {
        loading = document.createElement("div");
        loading.id = "loading";
        loading.innerHTML = "Đang xử lý... <span class='spinner'></span>";
        loading.style.cssText = "margin-top: 20px; font-size: 16px; color: #333;";
        document.body.appendChild(loading);
    }
    loading.style.display = "block";

    // Thêm CSS cho spinner
    const style = document.createElement("style");
    style.textContent = `
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
            display: inline-block;
            margin-left: 5px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    `;
    document.head.appendChild(style);

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch("/process-image", {
            method: "POST",
            body: formData
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ error: "Lỗi không xác định" }));
            throw new Error(err.error || "Không thể xử lý");
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
        resultImage.src = data.image_url + "?t=" + new Date().getTime(); // Tránh cache
    } catch (error) {
        alert("Lỗi: " + error.message);
    } finally {
        // Ẩn loading
        if (loading) loading.style.display = "none";
    }
};