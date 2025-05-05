// Main JavaScript for Car Essay Evaluator

document.addEventListener('DOMContentLoaded', function() {
    // Form submission - show loading indicator
    const evaluateForm = document.getElementById('evaluate-form');
    const submitButton = document.getElementById('submit-button');
    const loadingSpinner = document.getElementById('loading-spinner');
    
    if (evaluateForm) {
        evaluateForm.addEventListener('submit', function() {
            if (submitButton && loadingSpinner) {
                submitButton.disabled = true;
                submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Evaluating...';
                loadingSpinner.classList.remove('d-none');
            }
        });
    }
    
    // Word counter for essay textarea
    const essayTextarea = document.getElementById('id_content');
    const wordCountDisplay = document.getElementById('word-count');
    
    if (essayTextarea && wordCountDisplay) {
        essayTextarea.addEventListener('input', function() {
            const text = essayTextarea.value.trim();
            const wordCount = text ? text.split(/\s+/).length : 0;
            wordCountDisplay.textContent = wordCount;
            
            // Change color based on count
            if (wordCount < 50) {
                wordCountDisplay.className = 'text-danger';
            } else if (wordCount < 250) {
                wordCountDisplay.className = 'text-warning';
            } else {
                wordCountDisplay.className = 'text-success';
            }
        });
        
        // Trigger on page load
        essayTextarea.dispatchEvent(new Event('input'));
    }
    
    // Grade color adjustment on results page
    const gradeElement = document.getElementById('gradeContainer');
    if (gradeElement) {
        const grade = gradeElement.textContent.trim();
        if (grade.startsWith('A')) {
            gradeElement.classList.add('grade-a');
        } else if (grade.startsWith('B')) {
            gradeElement.classList.add('grade-b');
        } else if (grade.startsWith('C')) {
            gradeElement.classList.add('grade-c');
        } else if (grade.startsWith('D')) {
            gradeElement.classList.add('grade-d');
        } else {
            gradeElement.classList.add('grade-f');
        }
    }
    
    // Tooltips initialization
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Add highlighting to car-related keywords in essay content
    const essayContentElement = document.querySelector('.essay-content');
    if (essayContentElement) {
        highlightCarKeywords(essayContentElement);
    }
});

// Function to highlight car-related keywords
function highlightCarKeywords(element) {
    // List of car-related keywords to highlight
    const carKeywords = [
        'car', 'automobile', 'vehicle', 'engine', 'transmission', 'horsepower', 'torque',
        'sedan', 'suv', 'coupe', 'hatchback', 'convertible', 'wagon', 'truck', 'van',
        'electric', 'hybrid', 'gasoline', 'diesel', 'combustion', 'battery', 'charging',
        'brake', 'suspension', 'steering', 'wheel', 'tire', 'safety', 'airbag',
        'toyota', 'honda', 'ford', 'chevrolet', 'bmw', 'mercedes', 'audi', 'tesla',
        'volkswagen', 'hyundai', 'kia', 'nissan', 'mazda', 'subaru', 'lexus', 'porsche'
    ];
    
    // Technical terms
    const technicalTerms = [
        'engine', 'transmission', 'horsepower', 'torque', 'combustion', 'turbo', 'cylinder',
        'fuel', 'efficiency', 'brake', 'suspension', 'electrical', 'battery'
    ];
    
    // Environmental terms
    const environmentalTerms = [
        'electric', 'hybrid', 'emissions', 'carbon', 'eco', 'environment', 'sustainable'
    ];
    
    // Safety terms
    const safetyTerms = [
        'safety', 'airbag', 'crash', 'collision', 'autonomous', 'assist', 'braking'
    ];
    
    let content = element.innerHTML;
    
    // Highlight car keywords based on category
    function getTermCategory(term) {
        if (technicalTerms.includes(term.toLowerCase())) return 'technical';
        if (environmentalTerms.includes(term.toLowerCase())) return 'environmental';
        if (safetyTerms.includes(term.toLowerCase())) return 'safety';
        return '';
    }
    
    // Sort keywords by length (longest first) to avoid partial matches
    carKeywords.sort((a, b) => b.length - a.length);
    
    // Replace keywords with highlighted spans
    carKeywords.forEach(keyword => {
        const category = getTermCategory(keyword);
        const categoryClass = category ? ` ${category}` : '';
        
        // Using regex with word boundaries to match whole words only
        const regex = new RegExp(`\\b${keyword}\\b`, 'gi');
        content = content.replace(regex, `<span class="category-badge${categoryClass}">$&</span>`);
    });
    
    element.innerHTML = content;
}

// Function to format score as a grade
function formatScoreAsGrade(score) {
    if (score >= 9.5) return ["A+", "Outstanding"];
    if (score >= 9.0) return ["A", "Excellent"];
    if (score >= 8.5) return ["A-", "Very Good"];
    if (score >= 8.0) return ["B+", "Good"];
    if (score >= 7.5) return ["B", "Above Average"];
    if (score >= 7.0) return ["B-", "Slightly Above Average"];
    if (score >= 6.5) return ["C+", "Average"];
    if (score >= 6.0) return ["C", "Satisfactory"];
    if (score >= 5.5) return ["C-", "Below Average"];
    if (score >= 5.0) return ["D+", "Needs Improvement"];
    if (score >= 4.0) return ["D", "Poor"];
    if (score >= 3.0) return ["D-", "Very Poor"];
    return ["F", "Unsatisfactory"];
}

// Function to initialize radar chart for topic coverage
function initCoverageChart(chartId, labels, data) {
    const ctx = document.getElementById(chartId);
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Coverage',
                data: data,
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgb(54, 162, 235)',
                pointBackgroundColor: 'rgb(54, 162, 235)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgb(54, 162, 235)'
            }]
        },
        options: {
            elements: {
                line: {
                    borderWidth: 3
                }
            },
            scales: {
                r: {
                    beginAtZero: true
                }
            }
        }
    });
}
