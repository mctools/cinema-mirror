

#ifndef Prompt_GeoTree_hh
#define Prompt_GeoTree_hh

#include "PromptCore.hh"
#include <VecGeom/base/Transformation3D.h>

namespace Prompt {
  class GeoTree
  {
  public:
    struct Node {
      unsigned physical = -1, logical = -1;
      vecgeom::Transformation3D matrix;
      std::vector<std::shared_ptr<Node>> child;
      std::vector<unsigned> childPhysicalID;
      static std::vector<std::shared_ptr<Node>> allNodes;

      void print()
      {
        std::cout << "node physicalID " << physical << ", logicalID " << logical << "\n";
        matrix.Print();
        if(!childPhysicalID.empty())
        {
          std::cout << "\nChild physical ID: ";
          for (const auto &c : childPhysicalID)
          {
            std::cout << c << "  ";
          }
          std::cout << "\n";
        }

        if(!child.empty())
        {
          std::cout << "Child physical and logical ID (from the child objects):\n";
          for (const auto &c : child)
          {
            std::cout << "[" << c->physical << ", ";
            std::cout << c->logical << "],  ";
          }
          std::cout << "\n";
        }
        std::cout << "\n";
      }

      void printAllNodes()
      {
        for(auto node : allNodes)
          node->print();
      }

      void clearAllNodes()
      {
        allNodes.clear();
      }

      void addChild(std::shared_ptr<Node> c)
      {
        child.push_back(c);
      }

      void setMatrix(const vecgeom::Transformation3D *mat)
      {
        matrix = vecgeom::Transformation3D(* const_cast<vecgeom::Transformation3D*>(mat));
        matrix.Inverse(matrix);
      }
    };


  public:
    GeoTree();
    ~GeoTree();

    void print(bool phys=true);

    std::shared_ptr<Node> getRoot();

    std::vector<std::shared_ptr<Node>> findMotherNodeByPhysical(int num);

    std::shared_ptr<Node> findNodeByPhysical(int num);
    std::shared_ptr<Node> findNodeByPhysical(const std::shared_ptr<Node> &node, int num);

    std::vector<std::shared_ptr<Node>> findNodeByLogical(int num);
    void findNodeByLogical(const std::shared_ptr<Node> &node, int num, std::vector<std::shared_ptr<Node>>& logicalnode);


  private:
    std::shared_ptr<Node> m_root;
    void print(const std::shared_ptr<Node> &node, int layer, std::vector<std::vector<int>> &printArray, bool phys);

  };

}


#endif
