document.addEventListener('DOMContentLoaded', () => {
    console.log("✅ Directory loaded");

    // Default location → Hyderabad
    const map = L.map('map').setView([17.3850, 78.4867], 11);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
    }).addTo(map);

    let markers = [];

    // Load recyclers
    fetch('/directory_data')
        .then(res => res.json())
        .then(data => {
            renderRecyclers(data);
            addMarkers(data);
        })
        .catch(err => {
            console.error("❌ Failed to load recyclers:", err);
        });

    // Render recycler list
    function renderRecyclers(data) {
        const list = document.getElementById('recyclerList');
        list.innerHTML = '';

        if (!data || data.length === 0) {
            list.innerHTML =
                '<li class="p-3 text-gray-600">No recyclers found.</li>';
            return;
        }

        data.forEach(r => {
            const li = document.createElement('li');
            li.className =
                'p-3 bg-white border rounded hover:bg-green-50 cursor-pointer';

            li.innerHTML = `
                <h3 class="font-semibold text-lg">${r.name}</h3>
                <div class="mt-1 text-sm text-gray-600">
                    <div><strong>Type:</strong> ${r.type}</div>
                    <div><strong>City:</strong> ${r.city}</div>
                </div>
            `;

            // Click → focus on map
            li.addEventListener("click", () => {
                if (!r.lat || !r.lng) return;

                map.setView([r.lat, r.lng], 13);
                L.popup()
                    .setLatLng([r.lat, r.lng])
                    .setContent(`
                        <b>${r.name}</b><br>
                        ${r.city}<br>
                        Type: ${r.type}
                    `)
                    .openOn(map);
            });

            list.appendChild(li);
        });
    }

    // Add markers on map
    function addMarkers(data) {
        markers.forEach(m => map.removeLayer(m));
        markers = [];

        data.forEach(r => {
            if (!r.lat || !r.lng) return;

            const marker = L.marker([r.lat, r.lng]).addTo(map);
            marker.bindPopup(`
                <b>${r.name}</b><br>
                ${r.city}<br>
                Type: ${r.type}
            `);

            markers.push(marker);
        });
    }

    // Filter recyclers
    window.filterRecyclers = function () {
        const type = document.getElementById("filterType").value;
        const city = document.getElementById("filterCity").value;

        fetch(`/directory_data?type=${type}&city=${city}`)
            .then(res => res.json())
            .then(data => {
                renderRecyclers(data);
                addMarkers(data);
            })
            .catch(err => console.error("❌ Filter error:", err));
    };
});
