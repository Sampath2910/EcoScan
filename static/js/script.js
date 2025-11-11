// --- Global Variables ---
let allRecyclers = [];
let map;
let markers = [];
const DEFAULT_COORDS = { lat: 20.5937, lng: 78.9629 }; // Center of India

// --- Initialization ---

/**
 * Initializes the Google Map and fetches the recycler data.
 * This function is called via the 'callback' parameter in the Google Maps API script tag.
 */
function initMap() {
    // 1. Initialize the Map
    const mapElement = document.getElementById('map');
    if (mapElement) {
        map = new google.maps.Map(mapElement, {
            zoom: 5,
            center: DEFAULT_COORDS,
            mapId: 'RECYCLER_MAP_ID' // You can use a specific ID if you configure one in Google Cloud
        });
    }

    // 2. Fetch data from the Flask API endpoint
    fetchRecyclerData();
}

/**
 * Fetches the recycler data from the /directory_data Flask API.
 */
function fetchRecyclerData() {
    fetch('/directory_data')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            allRecyclers = data;
            // Initial rendering of the list and markers
            renderRecyclers(allRecyclers);
        })
        .catch(error => {
            console.error("Error fetching directory data:", error);
            const list = document.getElementById('recyclerList');
            if (list) {
                list.innerHTML = '<li class="error-message">Could not load recycling data.</li>';
            }
        });
}


// --- Rendering and Filtering Logic ---

/**
 * Renders the list items and map markers for a given list of recyclers.
 * @param {Array<Object>} recyclers - The list of recycler objects to display.
 */
function renderRecyclers(recyclers) {
    const list = document.getElementById('recyclerList');
    if (!list) return;

    list.innerHTML = ''; // Clear previous results
    clearMarkers();      // Clear previous markers from the map

    if (recyclers.length === 0) {
        list.innerHTML = '<li class="no-results">No recycling centers found matching your filters.</li>';
        return;
    }

    recyclers.forEach(recycler => {
        // 1. Add to List
        const listItem = document.createElement('li');
        listItem.className = 'recycler-item';
        listItem.innerHTML = `
            <h3>${recycler.name}</h3>
            <p><strong>Type:</strong> ${recycler.type}</p>
            <p><strong>City:</strong> ${recycler.city}</p>
        `;
        list.appendChild(listItem);

        // 2. Add to Map
        if (map && recycler.lat && recycler.lng) {
            const position = { lat: recycler.lat, lng: recycler.lng };
            const marker = new google.maps.Marker({
                position: position,
                map: map,
                title: recycler.name,
                icon: {
                    url: getIconColor(recycler.type), // Custom icon based on waste type
                    scaledSize: new google.maps.Size(30, 30)
                }
            });

            // Add info window
            const infoWindow = new google.maps.InfoWindow({
                content: `
                    <h4>${recycler.name}</h4>
                    <p>Type: ${recycler.type}<br>City: ${recycler.city}</p>
                `
            });
            marker.addListener('click', () => {
                infoWindow.open(map, marker);
            });

            markers.push(marker);
        }
    });

    // Optionally adjust the map bounds to fit the new markers
    if (markers.length > 0) {
        fitBoundsToMarkers();
    }
}

/**
 * Clears existing markers from the map.
 */
function clearMarkers() {
    markers.forEach(marker => marker.setMap(null));
    markers = [];
}

/**
 * Adjusts the map view to contain all current markers.
 */
function fitBoundsToMarkers() {
    const bounds = new google.maps.LatLngBounds();
    markers.forEach(marker => bounds.extend(marker.getPosition()));
    map.fitBounds(bounds);
}

/**
 * Determines the color of the map marker based on the waste type.
 * @param {string} type - The waste type (Plastic, Glass, etc.)
 * @returns {string} The URL for a colored marker icon.
 */
function getIconColor(type) {
    // Note: You would typically use actual custom marker images here.
    // For simplicity, we use a generic placeholder that Google Maps supports.
    switch (type.toLowerCase()) {
        case 'plastic':
            return 'http://maps.google.com/mapfiles/ms/icons/green-dot.png';
        case 'glass':
            return 'http://maps.google.com/mapfiles/ms/icons/blue-dot.png';
        case 'metal':
            return 'http://maps.google.com/mapfiles/ms/icons/yellow-dot.png';
        case 'paper':
        case 'cardboard':
            return 'http://maps.google.com/mapfiles/ms/icons/orange-dot.png';
        case 'trash':
            return 'http://maps.google.com/mapfiles/ms/icons/red-dot.png';
        default:
            return 'http://maps.google.com/mapfiles/ms/icons/purple-dot.png';
    }
}

/**
 * Filters the list of recyclers based on user input.
 * This function is called by the 'onchange' and 'oninput' events in directory.html.
 */
function filterRecyclers() {
    const typeFilter = document.getElementById('filterType').value.toLowerCase();
    const cityFilter = document.getElementById('filterCity').value.toLowerCase().trim();

    const filtered = allRecyclers.filter(recycler => {
        // Filter by Type
        const typeMatch = typeFilter === '' || recycler.type.toLowerCase() === typeFilter;

        // Filter by City (case-insensitive partial match)
        const cityMatch = cityFilter === '' || recycler.city.toLowerCase().includes(cityFilter);

        return typeMatch && cityMatch;
    });

    renderRecyclers(filtered);
}

// Ensure initMap is globally accessible if using the async/defer method
window.initMap = initMap;