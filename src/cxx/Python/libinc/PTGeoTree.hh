

#ifndef Prompt_GeoTree_hh
#define Prompt_GeoTree_hh

#include "PromptCore.hh"
#include <VecGeom/base/Transformation3D.h>

namespace Prompt {
  class GeoTree
  {
  public:
    struct Node {
      int physical = -1, logical = -1;
      vecgeom::Transformation3D *matrix = nullptr;
      std::vector<std::shared_ptr<Node>> child;

      void print()
      {
        std::cout << "node physicalID " << physical << ", logicalID " << logical << "\n";
        if (matrix)
          matrix->Print();
        if(child.empty())
          std::cout << "no physical child:\n";
        else
        {
          std::cout << "Child physical and logical ID:\n";
          for (const auto &c : child)
          {
            std::cout << "[" << c->physical << ", ";
            std::cout << c->logical << "],  ";
          }
          std::cout << "\n";
        }
      }

      void addChild(std::shared_ptr<Node> c)
      {
        child.push_back(c);
      }
    };

  public:
    GeoTree();
    ~GeoTree();

    void print();

    std::shared_ptr<Node> getRoot();
    std::shared_ptr<Node> findPhysicalChild(int num);
    std::shared_ptr<Node> findPhysicalChild(std::shared_ptr<Node> node, int num);

    std::vector<std::shared_ptr<Node>> findLogicalChild(int num);
    void findLogicalChild(std::shared_ptr<Node> node, int num, std::vector<std::shared_ptr<Node>>& logicalnode);


  private:
    std::shared_ptr<Node> m_root;
    void printNode(std::shared_ptr<Node> node, int layer, std::vector<std::vector<int>> &printArray);

  };

}


#endif
