TitikTemu TODO:
1. Map does not show on phone/tablet.
2. Remove sidebar collapse icon (fixable by me)
3. Map moves along the user scrolls down, or adjusted to the current user’s viewport height
4. Format insight to be more professional. Also reformat all grid_* to be the region name so that the user can easily recognize it
5. Sidebar collapsed the map is gray/blank on the right side
6. User’s location does not show in map
7. Update Self Tracker fields to accept more input relevant to the fields that titiktemu-analytics use to process pipeline
8. Button on-off toggle bug in profil usaha umkm user
9. Logo blur (fixable by me)
10. Add view reallocation for the user if the current user location is bahaya
11. User’s location does not show in map
12. Make the current user’s location as red and give pulse animation, and add also for the reallocated place user’s get
13. Display the grid more user-friendly. Dont display the training grid, just draw a circle zone with red, green, or yellow colour depending on the grid ews score
14. User who wants to reallocate can fill a form and select reallocation place that get sends to the operator laporan alokasi
15. Smart tenant matching engine for operator still showing waspada, even though it should have been bahaya only
Each UMKM in Operator discovery map when clicked, yes it gets redirected to the place in map, but still does not show a dot/point indicating that they are there
16. When user is logged in to the beranda page, immediately asked for prompt to access their location, and then processes it to show display it in the map, and the given coordinates will apply also for another page that displaying a map

TODO-v2:
1. The Reallocation Report for operator Laporan Alokasi page is failed or error 401 unauthorized (even though i already authorized) and the response is Invalid or expired token
2. The Beranda Map is still not following the user's viewport height when scroll, the map stay still in the top (bad UX), applies to all the page using map
3. There are still many grid_* text, make sure dont expose this to the UMKM user and operator, use a clear and recognizable label
4. Fix ESG Dashboard filter in Operator, the filter does not work as it should, when the operator clicks and filter a spesific region, the dashboard charts will be automatically display that region only. Also make the charts/graph to have more interactiveness
5. In the Beranda page, make sure to focus/zoom more on the current user's location so that the user did not have to search for their dots in the map.
6. Fix all the necessary UI components, e.g. double icon for Alert, etc. Make sure all good.
7. When the user succesfully update/fill the Profil Usaha form, it automatically changes the current user coordinates in the map. And after updating the data/submitting the form, the UMKM Self Tracker page becomes a informative table/dashboard with a necessary metrics for the current user's location and region.
8. If the user is outside the study grid, display a good message e.g. return to beranda or else,indicating they does not know.
9. If the UMKM User current location is in the red grid, display the pulse dot points in the map as red, applies to the other yellow (waspada) and green (aman)
10. Differentiate Panel Informasi information between UMKM user and Operator. For the UMKM user, just give statistics around the user's region, not the whole study grid (this is for operator). And also make sure Rekomendasi Alokasi is different for each UMKM User based on their location (prioritize nearest allocation if exist).
11. For Operator Discovery Map, add a pulse for each selected/observed UMKM
12. Adjust positioing of legend and AI Chatbot in the Beranda page for both UMKM user and Operator. For Operator Discovery Map page, place the AI Chatbot stick to the map. And the legend close bottom near the map so user dont have to
13. For Operator Smart Tenant Matching, adjust all the layout to be more user friendly. And remove Tolak/Terima button because now it belongs to Laporan Alokasi job.
14. Remove n=119 for all page, just show confidence/accuracy score
15. Format all UMKM name to be capitalized for each beginning letter e.g. aa fun chicken -> AA Fun Chicken, auntie annes -> Auntie Annes, etc.
16. Adjust all mobile/tablet layout for all page both operator and umkm user, make sure all responsive, e.g. in mobile viewport for operator tenant matching, the legend and the AI Chatbot is panel is too big and covers the map.
17. If the UMKM user is on the red zone, for every Rekomendasi Alokasi in the Beranda page, add View Reallocation Button that redirects to the Lihat Realokasi state (immediately shows the selected reallocation candidate in map) along with the necessary detail
18. Dont forget to add a pulse animation for every reallocated candidates, current user's location, etc.
