/**
 * GraphRAG - Main JavaScript functionality
 */

// Initialize tooltips and popovers
document.addEventListener('DOMContentLoaded', function() {
  // Initialize Bootstrap tooltips
  const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
  tooltipTriggerList.map(function(tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
  });

  // Initialize Bootstrap popovers
  const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
  popoverTriggerList.map(function(popoverTriggerEl) {
    return new bootstrap.Popover(popoverTriggerEl);
  });

  // Initialize dynamic components
  initializeDynamicComponents();
});

/**
 * Initialize dynamic components throughout the application
 */
function initializeDynamicComponents() {
  // Initialize entity highlighting
  highlightEntities();
  
  // Initialize tag input
  initializeTagInput();
  
  // Initialize file upload preview
  initializeFileUpload();
  
  // Initialize search form
  initializeSearchForm();
  
  // Initialize multi-hop reasoning toggle
  initializeMultiHopToggle();
}

/**
 * Highlight entities in text
 */
function highlightEntities() {
  const entityContainers = document.querySelectorAll('.entity-container');
  
  entityContainers.forEach(container => {
    const text = container.textContent;
    const entities = JSON.parse(container.getAttribute('data-entities') || '[]');
    
    if (entities.length === 0) return;
    
    // Sort entities by their start position in reverse order
    // (to avoid messing up the indices when adding spans)
    entities.sort((a, b) => b.start - a.start);
    
    let highlightedText = text;
    
    entities.forEach(entity => {
      const entityStart = entity.start;
      const entityEnd = entity.end;
      const entityType = entity.type || 'DEFAULT';
      const entityName = text.substring(entityStart, entityEnd);
      
      const before = highlightedText.substring(0, entityStart);
      const after = highlightedText.substring(entityEnd);
      
      const entitySpan = `<span class="entity-highlight entity-${entityType}" 
                              data-entity-type="${entityType}"
                              data-entity-id="${entity.id || ''}"
                              data-bs-toggle="tooltip"
                              title="${entityType}">${entityName}</span>`;
                              
      highlightedText = before + entitySpan + after;
    });
    
    container.innerHTML = highlightedText;
  });
}

/**
 * Initialize tag input for document upload/edit
 */
function initializeTagInput() {
  const tagInput = document.getElementById('tag-input');
  const tagContainer = document.getElementById('tag-container');
  const hiddenTagInput = document.getElementById('tags');
  
  if (!tagInput || !tagContainer || !hiddenTagInput) return;
  
  // Initialize with existing tags if any
  const existingTags = hiddenTagInput.value.split(',').filter(tag => tag.trim() !== '');
  existingTags.forEach(tag => addTag(tag.trim()));
  
  tagInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      
      const tag = this.value.trim();
      if (tag) {
        addTag(tag);
        this.value = '';
        updateHiddenInput();
      }
    }
  });
  
  // Handle clicks on tag removal buttons
  tagContainer.addEventListener('click', function(e) {
    if (e.target.classList.contains('tag-remove')) {
      e.target.parentElement.remove();
      updateHiddenInput();
    }
  });
  
  function addTag(text) {
    // Check if tag already exists
    const existingTags = tagContainer.querySelectorAll('.tag-text');
    for (let i = 0; i < existingTags.length; i++) {
      if (existingTags[i].textContent.toLowerCase() === text.toLowerCase()) {
        return;
      }
    }
    
    const tag = document.createElement('span');
    tag.className = 'badge bg-primary me-2 mb-2';
    tag.innerHTML = `<span class="tag-text">${text}</span> <span class="tag-remove" role="button">&times;</span>`;
    tagContainer.appendChild(tag);
  }
  
  function updateHiddenInput() {
    const tags = Array.from(tagContainer.querySelectorAll('.tag-text')).map(el => el.textContent);
    hiddenTagInput.value = tags.join(',');
  }
}

/**
 * Initialize file upload preview
 */
function initializeFileUpload() {
  const fileInput = document.getElementById('file');
  const filePreview = document.getElementById('file-preview');
  const fileLabel = document.querySelector('label[for="file"]');
  
  if (!fileInput || !filePreview) return;
  
  fileInput.addEventListener('change', function() {
    if (this.files && this.files[0]) {
      const file = this.files[0];
      
      // Update label
      if (fileLabel) {
        fileLabel.textContent = file.name;
      }
      
      // Show file info in preview
      let fileIcon = 'file-text';
      let fileType = file.type.split('/')[1] || 'unknown';
      
      if (file.type.startsWith('image/')) {
        fileIcon = 'image';
      } else if (file.type === 'application/pdf') {
        fileIcon = 'file-pdf';
      } else if (file.type.includes('word') || file.type.includes('document')) {
        fileIcon = 'file-word';
      }
      
      const fileSize = formatFileSize(file.size);
      
      filePreview.innerHTML = `
        <div class="card bg-dark">
          <div class="card-body d-flex align-items-center">
            <div class="file-icon file-icon-${fileType}">
              <i class="feather icon-${fileIcon}"></i>
            </div>
            <div>
              <h5 class="card-title">${file.name}</h5>
              <p class="card-text text-secondary">${fileType.toUpperCase()} - ${fileSize}</p>
            </div>
          </div>
        </div>
      `;
      
      filePreview.classList.remove('d-none');
    }
  });
}

/**
 * Format file size in human-readable format
 */
function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Initialize search form behaviors
 */
function initializeSearchForm() {
  const searchForm = document.getElementById('search-form');
  const searchInput = document.getElementById('search-input');
  const searchResultsContainer = document.getElementById('search-results');
  const loadingIndicator = document.getElementById('search-loading');
  
  if (!searchForm) return;
  
  searchForm.addEventListener('submit', function(e) {
    if (searchInput && searchInput.value.trim() === '') {
      e.preventDefault();
      // Focus on the input
      searchInput.focus();
      return;
    }
    
    // Show loading indicator if it exists
    if (loadingIndicator) {
      loadingIndicator.classList.remove('d-none');
    }
    
    // Hide results while loading
    if (searchResultsContainer) {
      searchResultsContainer.classList.add('d-none');
    }
  });
}

/**
 * Initialize multi-hop reasoning toggle
 */
function initializeMultiHopToggle() {
  const multiHopToggle = document.getElementById('multi-hop-toggle');
  const multiHopInfo = document.getElementById('multi-hop-info');
  
  if (!multiHopToggle || !multiHopInfo) return;
  
  multiHopToggle.addEventListener('change', function() {
    if (this.checked) {
      multiHopInfo.classList.remove('d-none');
    } else {
      multiHopInfo.classList.add('d-none');
    }
  });
}

/**
 * Handle entity click in document text
 */
function handleEntityClick(entityId) {
  if (!entityId) return;
  
  const modal = new bootstrap.Modal(document.getElementById('entity-modal'));
  
  // Show loading state
  document.getElementById('entity-modal-content').innerHTML = `
    <div class="loader-container">
      <div class="loader"></div>
    </div>
  `;
  
  // Show the modal
  modal.show();
  
  // Fetch entity data
  fetch(`/api/knowledge-graph/entities/${entityId}`)
    .then(response => response.json())
    .then(data => {
      // Update modal content
      const entityConnections = data.connections || [];
      let modalContent = '';
      
      if (entityConnections.length === 0) {
        modalContent = '<p class="text-center">No connections found for this entity.</p>';
      } else {
        // Group by relationship type
        const relationshipGroups = {};
        entityConnections.forEach(conn => {
          if (!relationshipGroups[conn.relationship]) {
            relationshipGroups[conn.relationship] = [];
          }
          relationshipGroups[conn.relationship].push(conn);
        });
        
        // Build the content
        modalContent = '<div class="list-group">';
        
        for (const [relationship, connections] of Object.entries(relationshipGroups)) {
          modalContent += `<h6 class="mt-3">${formatRelationshipName(relationship)} (${connections.length})</h6>`;
          
          connections.forEach(conn => {
            const confidenceClass = getConfidenceClass(conn.confidence);
            modalContent += `
              <div class="list-group-item bg-dark">
                <div class="d-flex justify-content-between align-items-center">
                  <div>
                    <span class="badge bg-info me-2">${conn.entity_type}</span>
                    <strong>${conn.name}</strong>
                  </div>
                  <span class="badge ${confidenceClass}">${Math.round(conn.confidence * 100)}%</span>
                </div>
              </div>
            `;
          });
        }
        
        modalContent += '</div>';
      }
      
      document.getElementById('entity-modal-content').innerHTML = modalContent;
    })
    .catch(error => {
      console.error('Error fetching entity data:', error);
      document.getElementById('entity-modal-content').innerHTML = `
        <div class="alert alert-danger">
          Error loading entity information.
        </div>
      `;
    });
}

/**
 * Format relationship name for display
 */
function formatRelationshipName(name) {
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, l => l.toUpperCase());
}

/**
 * Get bootstrap badge class based on confidence value
 */
function getConfidenceClass(confidence) {
  if (confidence >= 0.8) {
    return 'bg-success';
  } else if (confidence >= 0.6) {
    return 'bg-info';
  } else if (confidence >= 0.4) {
    return 'bg-warning';
  } else {
    return 'bg-secondary';
  }
}

/**
 * Show visualization of a reasoning path
 */
function showReasoningPath(pathIndex, resultIndex) {
  const result = document.getElementById(`search-result-${resultIndex}`);
  if (!result) return;
  
  const pathData = result.getAttribute(`data-reasoning-path-${pathIndex}`);
  if (!pathData) return;
  
  const path = JSON.parse(pathData);
  
  const modalTitle = document.getElementById('reasoning-path-modal-title');
  const modalContent = document.getElementById('reasoning-path-modal-content');
  
  if (!modalTitle || !modalContent) return;
  
  // Set title
  modalTitle.textContent = `Reasoning Path #${pathIndex + 1}`;
  
  // Build path visualization
  let pathHtml = '<div class="reasoning-path">';
  
  path.forEach((step, index) => {
    pathHtml += `
      <div class="reasoning-path-step">
        <div>
          <span class="badge bg-info">${step.source.entity_type}</span>
          <strong>${step.source.name}</strong>
        </div>
        <div class="reasoning-path-arrow">
          <i class="feather icon-arrow-right"></i>
          <div class="small text-secondary">${formatRelationshipName(step.relationship)} (${Math.round(step.confidence * 100)}%)</div>
        </div>
        <div>
          <span class="badge bg-info">${step.target.entity_type}</span>
          <strong>${step.target.name}</strong>
        </div>
      </div>
    `;
    
    // Add separator if not the last step
    if (index < path.length - 1) {
      pathHtml += '<hr class="my-2">';
    }
  });
  
  pathHtml += '</div>';
  
  modalContent.innerHTML = pathHtml;
  
  // Show the modal
  const modal = new bootstrap.Modal(document.getElementById('reasoning-path-modal'));
  modal.show();
}
