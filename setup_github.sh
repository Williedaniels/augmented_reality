#!/bin/bash

# Augmented Reality - GitHub Setup Script
# This script helps you quickly set up the repository for GitHub

echo "🎨 Augmented Reality - GitHub Setup"
echo "=================================="

# Check if git is available
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install git first."
    exit 1
fi

# Initialize git repository
echo "📁 Initializing git repository..."
git init

# Add all files
echo "📝 Adding files to git..."
git add .

# Create initial commit
echo "💾 Creating initial commit..."
git commit -m "Initial commit: Complete Augmented Reality system with Python and OpenCV

Features:
- Marker-based AR with real-time detection
- Feature detection with ORB, SIFT, SURF
- Robust homography estimation with RANSAC
- 3D model projection (cube, pyramid, axes)
- Real-time camera processing and interactive controls
- Customizable markers and modular architecture

Built from scratch following build-your-own-x guidelines."

echo ""
echo "✅ Repository initialized successfully!"
echo ""
echo "🚀 Next steps:"
echo "1. Create a new repository on GitHub"
echo "2. Run: git remote add origin https://github.com/yourusername/your-repo-name.git"
echo "3. Run: git branch -M main"
echo "4. Run: git push -u origin main"
echo ""
echo "🎯 To test the AR system (no camera needed):"
echo "   python examples/test_ar_system.py"
echo ""
echo "📷 To run the real-time AR application:"
echo "   python src/ar_app.py"
echo ""

