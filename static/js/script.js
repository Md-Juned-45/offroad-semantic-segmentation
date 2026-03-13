// Define the exact classes from the hacking challenge
const CLASS_MAP = [
    { id: 0, name: "Background", color: "rgb(0, 0, 0)" },
    { id: 100, name: "Trees", color: "rgb(34, 139, 34)" },
    { id: 200, name: "Lush Bushes", color: "rgb(50, 205, 50)" },
    { id: 300, name: "Dry Grass", color: "rgb(218, 165, 32)" },
    { id: 500, name: "Dry Bushes", color: "rgb(184, 134, 11)" },
    { id: 550, name: "Ground Clutter", color: "rgb(160, 82, 45)" },
    { id: 600, name: "Flowers", color: "rgb(255, 20, 147)" },
    { id: 700, name: "Logs", color: "rgb(139, 69, 19)" },
    { id: 800, name: "Rocks", color: "rgb(128, 128, 128)" },
    { id: 7100, name: "Landscape", color: "rgb(210, 180, 140)" },
    { id: 10000, name: "Sky", color: "rgb(135, 206, 235)" }
];

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const resultsViewer = document.getElementById('results-viewer');
    const originalImage = document.getElementById('original-image');
    const maskImage = document.getElementById('mask-image');
    const imageContainer = document.getElementById('image-container');
    const slider = document.getElementById('slider');
    const runBtn = document.getElementById('run-btn');
    const newUploadBtn = document.getElementById('new-upload-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const statRes = document.getElementById('stat-res');
    const statTime = document.getElementById('stat-time');
    const viewBtns = document.querySelectorAll('.view-btn');
    
    let currentImageFile = null;
    let isDragging = false;
    let currentViewMode = 'split'; // split, overlay, mask

    // Initialize Legend
    const legendGrid = document.getElementById('legend-grid');
    CLASS_MAP.forEach(cls => {
        const item = document.createElement('div');
        item.className = 'legend-item';
        item.innerHTML = `
            <div class="color-box" style="background-color: ${cls.color}"></div>
            <span class="class-name">${cls.name}</span>
            <span class="class-id">${cls.id}</span>
        `;
        legendGrid.appendChild(item);
    });

    // --- Drag and Drop Handling ---
    dropZone.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dts = e.dataTransfer;
        const files = dts.files;
        handleFiles(files);
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length === 0) return;
        const file = files[0];
        
        if (!file.type.startsWith('image/')) {
            showToast('Please upload an image file.', 'error');
            return;
        }

        currentImageFile = file;
        const url = URL.createObjectURL(file);
        
        originalImage.src = url;
        maskImage.src = ''; // Clear previous mask
        maskImage.style.opacity = '0'; // Hide mask initially
        
        originalImage.onload = () => {
            statRes.innerText = `${originalImage.naturalWidth} x ${originalImage.naturalHeight}`;
        };
        
        dropZone.classList.add('hidden');
        resultsViewer.classList.remove('hidden');
        runBtn.disabled = false;
        statTime.innerText = '-- ms';
        
        // Reset View Controls
        resetViewControls();
    }

    // --- Switch Images ---
    newUploadBtn.addEventListener('click', () => {
        resultsViewer.classList.add('hidden');
        dropZone.classList.remove('hidden');
        currentImageFile = null;
        runBtn.disabled = true;
        maskImage.src = '';
    });

    // --- API Request ---
    runBtn.addEventListener('click', async () => {
        if (!currentImageFile) return;

        const formData = new FormData();
        formData.append('image', currentImageFile);

        loadingOverlay.classList.remove('hidden');
        runBtn.disabled = true;
        maskImage.style.opacity = '0';
        
        const startTime = performance.now();

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Server error');
            }

            const blob = await response.blob();
            const maskUrl = URL.createObjectURL(blob);
            
            maskImage.src = maskUrl;
            
            // Show mask smoothly
            maskImage.onload = () => {
                const endTime = performance.now();
                statTime.innerText = `${(endTime - startTime).toFixed(0)} ms`;
                
                loadingOverlay.classList.add('hidden');
                runBtn.disabled = false;
                
                // Show mask according to current view mode
                setViewMode(currentViewMode);
                showToast('Inference completed successfully', 'success');
            };

        } catch (error) {
            console.error(error);
            loadingOverlay.classList.add('hidden');
            runBtn.disabled = false;
            showToast(`Error: ${error.message}`, 'error');
        }
    });

    // --- View Navigation ---
    viewBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            setViewMode(e.currentTarget.dataset.view);
            
            viewBtns.forEach(b => b.classList.remove('active'));
            e.currentTarget.classList.add('active');
        });
    });

    function resetViewControls() {
        setViewMode('split');
        viewBtns.forEach(b => b.classList.remove('active'));
        document.querySelector('[data-view="split"]').classList.add('active');
    }

    function setViewMode(mode) {
        currentViewMode = mode;
        imageContainer.className = `image-container view-${mode}`;
        
        if (mode === 'split') {
            slider.style.left = '50%';
            maskImage.style.clipPath = `inset(0 0 0 50%)`;
            maskImage.style.opacity = '1';
        } else if (mode === 'overlay') {
            maskImage.style.clipPath = 'none';
            maskImage.style.opacity = '0.6';
        } else if (mode === 'mask') {
            maskImage.style.clipPath = 'none';
            maskImage.style.opacity = '1';
        }
    }

    // --- Slider Logic ---
    function moveSlider(clientX) {
        if (!isDragging || currentViewMode !== 'split') return;

        const rect = imageContainer.getBoundingClientRect();
        let x = clientX - rect.left;
        
        // Boundaries
        x = Math.max(0, Math.min(x, rect.width));
        
        const percent = (x / rect.width) * 100;
        
        // Move slider handle
        slider.style.left = `${percent}%`;
        
        // Clip the mask image (reveals right side of image)
        maskImage.style.clipPath = `inset(0 0 0 ${percent}%)`;
    }

    imageContainer.addEventListener('mousedown', (e) => {
        if (currentViewMode === 'split') {
            isDragging = true;
            moveSlider(e.clientX);
        }
    });

    window.addEventListener('mouseup', () => {
        isDragging = false;
    });

    window.addEventListener('mousemove', (e) => {
        moveSlider(e.clientX);
    });

    // Touch support for slider
    imageContainer.addEventListener('touchstart', (e) => {
        if (currentViewMode === 'split') {
            isDragging = true;
            moveSlider(e.touches[0].clientX);
        }
    });
    
    window.addEventListener('touchend', () => { isDragging = false; });
    window.addEventListener('touchmove', (e) => { moveSlider(e.touches[0].clientX); });


    // --- Toast Notification System ---
    function showToast(message, type = 'success') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icon = type === 'success' ? 'ri-check-line' : 'ri-error-warning-line';
        
        toast.innerHTML = `
            <i class="${icon}"></i>
            <span>${message}</span>
        `;
        
        container.appendChild(toast);
        
        setTimeout(() => {
            toast.style.animation = 'slideIn 0.3s backwards reverse';
            setTimeout(() => {
                container.removeChild(toast);
            }, 300);
        }, 3000);
    }
});
