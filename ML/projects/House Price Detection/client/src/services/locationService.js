import locationsData from '../data/locationsData.json';

/**
 * Service for querying Indian administrative and geographic hierarchy
 */
export const locationService = {
  /**
   * Returns list of all supported Indian States
   */
  getStates() {
    return locationsData.states || [];
  },

  /**
   * Returns districts belonging to a specific state (by state ID or name)
   */
  getDistricts(stateIdentifier) {
    if (!stateIdentifier) return [];
    
    let stateId = null;
    const strId = String(stateIdentifier).toLowerCase();
    const matchedState = (locationsData.states || []).find(
      (s) => String(s.id).toLowerCase() === strId ||
             (s.name && s.name.toLowerCase() === strId) ||
             (s.code && s.code.toLowerCase() === strId)
    );
    if (matchedState) {
      stateId = matchedState.id;
    } else if (!isNaN(Number(stateIdentifier))) {
      stateId = Number(stateIdentifier);
    }

    if (stateId === null) return [];
    return (locationsData.districts || []).filter((d) => d.state_id === stateId);
  },

  /**
   * Returns subdistricts / mandals belonging to a specific district
   */
  getMandals(districtIdentifier) {
    if (!districtIdentifier) return [];

    let distNumId = null;
    const strId = String(districtIdentifier).toLowerCase();
    const matchedDist = (locationsData.districts || []).find(
      (d) => String(d.id).toLowerCase() === strId ||
             String(d.district_id).toLowerCase() === strId ||
             (d.name && d.name.toLowerCase() === strId)
    );

    if (matchedDist) {
      distNumId = matchedDist.district_id;
    } else if (!isNaN(Number(districtIdentifier))) {
      distNumId = Number(districtIdentifier);
    }

    if (distNumId === null) return [];
    return (locationsData.mandals || []).filter((m) => m.district_id === distNumId);
  },

  /**
   * Returns villages / localities belonging to a specific mandal / subdistrict
   */
  getVillages(mandalIdentifier) {
    if (!mandalIdentifier) return [];

    let subdistNumId = null;
    const strId = String(mandalIdentifier).toLowerCase();
    const matchedMandal = (locationsData.mandals || []).find(
      (m) => String(m.id).toLowerCase() === strId ||
             String(m.subdistrict_id).toLowerCase() === strId ||
             (m.name && m.name.toLowerCase() === strId)
    );

    if (matchedMandal) {
      subdistNumId = matchedMandal.subdistrict_id;
    } else if (!isNaN(Number(mandalIdentifier))) {
      subdistNumId = Number(mandalIdentifier);
    }

    if (subdistNumId === null) return [];
    return (locationsData.localities || []).filter((l) => l.subdistrict_id === subdistNumId);
  },

  /**
   * Returns all localities with coordinates for map rendering
   */
  getAllLocalities() {
    return locationsData.localities || [];
  },

  /**
   * Returns localities belonging to a specific state (by state name or ID)
   */
  getLocalitiesByState(stateIdentifier) {
    if (!stateIdentifier) return [];
    return (locationsData.localities || []).filter(
      (l) => l.state_name && l.state_name.toLowerCase() === String(stateIdentifier).toLowerCase()
    );
  },

  /**
   * Returns localities belonging to a specific district (by district name or ID)
   */
  getLocalitiesByDistrict(districtIdentifier) {
    if (!districtIdentifier) return [];
    return (locationsData.localities || []).filter(
      (l) => l.district_name && l.district_name.toLowerCase() === String(districtIdentifier).toLowerCase()
    );
  },

  /**
   * Finds a locality by name and optionally district/state
   */
  findLocalityByName(localityName) {
    if (!localityName) return null;
    return (locationsData.localities || []).find(
      (l) => l.name.toLowerCase() === localityName.toLowerCase()
    );
  },
};

export default locationService;
