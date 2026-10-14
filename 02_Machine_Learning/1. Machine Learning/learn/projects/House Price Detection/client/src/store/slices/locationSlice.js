import { createSlice } from '@reduxjs/toolkit';
import locationService from '../../services/locationService';

const initialState = {
  states: locationService.getStates(),
  districts: [],
  mandals: [],
  villages: [],

  selectedState: null,
  selectedDistrict: null,
  selectedMandal: null,
  selectedVillage: null,

  loading: false,
  error: null,
};

export const locationSlice = createSlice({
  name: 'location',
  initialState,
  reducers: {
    setSelectedState: (state, action) => {
      const stateObj = action.payload;
      state.selectedState = stateObj;
      // Cascading reset
      state.selectedDistrict = null;
      state.selectedMandal = null;
      state.selectedVillage = null;
      state.districts = stateObj ? locationService.getDistricts(stateObj.id || stateObj.name) : [];
      state.mandals = [];
      state.villages = [];
    },
    setSelectedDistrict: (state, action) => {
      const distObj = action.payload;
      state.selectedDistrict = distObj;
      // Cascading reset
      state.selectedMandal = null;
      state.selectedVillage = null;
      state.mandals = distObj ? locationService.getMandals(distObj.district_id || distObj.id || distObj.name) : [];
      state.villages = [];
    },
    setSelectedMandal: (state, action) => {
      const mandalObj = action.payload;
      state.selectedMandal = mandalObj;
      // Cascading reset
      state.selectedVillage = null;
      state.villages = mandalObj ? locationService.getVillages(mandalObj.subdistrict_id || mandalObj.id || mandalObj.name) : [];
    },
    setSelectedVillage: (state, action) => {
      state.selectedVillage = action.payload;
    },
    // Allows setting entire hierarchy directly (e.g. from map click reverse-geocoding)
    setFullLocationHierarchy: (state, action) => {
      const { stateName, districtName, mandalName, villageName, localityObj } = action.payload;
      
      const foundState = state.states.find(
        (s) => s.name.toLowerCase() === (stateName || '').toLowerCase()
      );
      if (foundState) {
        state.selectedState = foundState;
        state.districts = locationService.getDistricts(foundState.id);
      }

      if (districtName) {
        const dClean = districtName.toLowerCase().replace(/ district| rural| urban/g, '').trim();
        const foundDist = state.districts.find(
          (d) => d.name.toLowerCase() === districtName.toLowerCase() ||
                 d.name.toLowerCase().includes(dClean) ||
                 dClean.includes(d.name.toLowerCase())
        );
        if (foundDist) {
          state.selectedDistrict = foundDist;
          state.mandals = locationService.getMandals(foundDist.district_id || foundDist.id || foundDist.name);
        }
      }

      if (mandalName && state.mandals.length > 0) {
        const mClean = mandalName.toLowerCase().replace(/ mandal| taluk| tehsil| block| urban| rural/g, '').trim();
        const foundMandal = state.mandals.find(
          (m) => m.name.toLowerCase() === mandalName.toLowerCase() ||
                 m.name.toLowerCase().includes(mClean) ||
                 mClean.includes(m.name.toLowerCase())
        );
        if (foundMandal) {
          state.selectedMandal = foundMandal;
          state.villages = locationService.getVillages(foundMandal.subdistrict_id || foundMandal.id || foundMandal.name);
        }
      }

      if (villageName || localityObj) {
        const foundLoc = localityObj || state.villages.find(
          (v) => v.name.toLowerCase() === (villageName || '').toLowerCase()
        );
        if (foundLoc) {
          state.selectedVillage = foundLoc;
        }
      }
    },
    clearLocation: (state) => {
      state.selectedState = null;
      state.selectedDistrict = null;
      state.selectedMandal = null;
      state.selectedVillage = null;
      state.districts = [];
      state.mandals = [];
      state.villages = [];
    },
  },
});

export const {
  setSelectedState,
  setSelectedDistrict,
  setSelectedMandal,
  setSelectedVillage,
  setFullLocationHierarchy,
  clearLocation,
} = locationSlice.actions;

export default locationSlice.reducer;
