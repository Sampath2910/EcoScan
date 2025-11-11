document.addEventListener('DOMContentLoaded', () => {
    console.log("✅ DOM loaded, initializing Leaflet map...");

    const mapContainer = document.getElementById('map');
    if (!mapContainer) {
        console.error("❌ Map container not found!");
        return;
    }

    // Initialize the map
    const map = L.map('map').setView([20.5937, 78.9629], 5);

    // Add OpenStreetMap tiles
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
    }).addTo(map);

    console.log("✅ Leaflet map initialized successfully!");

    let allRecyclers = [];
    let markers = [];

    // Fetch data
    fetch('/directory_data')
        .then(res => res.json())
        .then(data => {
            allRecyclers = data;
            renderRecyclers(data);
            addMarkers(data);
        })
        .catch(err => {
            console.error("❌ Error fetching recyclers:", err);
        });

    function renderRecyclers(data) {
        const list = document.getElementById('recyclerList');
        list.innerHTML = '';
        if (data.length === 0) {
            list.innerHTML = '<li class="p-3 text-gray-600">No recyclers found.</li>';
            return;
        }

        data.forEach(r => {
            const li = document.createElement('li');
            li.className = 'p-3 bg-white border rounded hover:bg-green-50 cursor-pointer';
            li.innerHTML = `
                <h3 class="font-semibold text-lg">${r.name}</h3>
                <p class="text-sm text-gray-600 capitalize">Type: ${r.type}</p>
                <p class="text-sm text-gray-600">City: ${r.city}</p>
            `;
            li.onclick = () => {
                map.setView([r.lat, r.lng], 12);
                L.popup()
                    .setLatLng([r.lat, r.lng])
                    .setContent(`<b>${r.name}</b><br>${r.city}<br>Type: ${r.type}`)
                    .openOn(map);
            };
            list.appendChild(li);
        });
    }

    function addMarkers(data) {
        markers.forEach(m => map.removeLayer(m));
        markers = [];

        data.forEach(r => {
            if (!r.lat || !r.lng) return;
            const marker = L.marker([r.lat, r.lng]).addTo(map);
            marker.bindPopup(`<b>${r.name}</b><br>${r.city}<br>Type: ${r.type}`);
            markers.push(marker);
        });
    }

    window.filterRecyclers = function () {
        const type = document.getElementById('filterType').value.toLowerCase();
        const city = document.getElementById('filterCity').value.toLowerCase();

        const filtered = allRecyclers.filter(r => {
            const matchesType = !type || r.type.toLowerCase() === type;
            const matchesCity = !city || r.city.toLowerCase().includes(city);
            return matchesType && matchesCity;
        });

        renderRecyclers(filtered);
        addMarkers(filtered);
    };
});
